import base64
import io
import inspect
import os
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import torch
from diffusers import ZImagePipeline
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

BASE_DIR = Path(os.getenv("COMFYUI_BASE_DIR", "/data1/w00916456/ComfyUI"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", str(BASE_DIR / "results"))).resolve()
QWEN_MODEL_PATH = os.getenv("QWEN_MODEL_PATH", "/data1/w00916456/Qwen3-VL/Qwen3-VL-32B-Instruct")
ZIMAGE_MODEL_PATH = os.getenv("ZIMAGE_MODEL_PATH", "/data1/w00916456/Z-Image-main/Z-Image-Turbo")
DEFAULT_QWEN_GPU_ID = int(os.getenv("DEFAULT_QWEN_GPU_ID", "0"))
DEFAULT_ZIMAGE_GPU_ID = int(os.getenv("DEFAULT_ZIMAGE_GPU_ID", "0"))

app = FastAPI(title="Result1 Inference API", version="0.2.0")

_state: Dict[str, Any] = {
    "qwen_model": None,
    "processor": None,
    "qwen_device": None,
    "zimage_pipe": None,
    "zimage_device": None,
}
_infer_lock = Lock()


class InferPathRequest(BaseModel):
    image_path: str
    user_input: str = Field(default="Describe this image for re-generation.")
    min_pixels: int = 117600
    max_pixels: int = 786432
    max_new_tokens: int = 1024
    qwen_gpu_id: Optional[int] = None
    zimage_gpu_id: Optional[int] = None

    seed: Optional[int] = None
    prompt_override: Optional[str] = None
    negative_prompt: Optional[str] = None
    guidance_scale: float = 1.0
    num_inference_steps: int = 4
    width: Optional[int] = None
    height: Optional[int] = None


def _ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_qwen_model(device: str) -> None:
    if _state["qwen_model"] is not None and _state["qwen_device"] == device:
        return
    _state["qwen_model"] = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_PATH,
        torch_dtype="auto",
        device_map=device,
    )
    _state["processor"] = AutoProcessor.from_pretrained(QWEN_MODEL_PATH)
    _state["qwen_device"] = device


def _load_zimage_model(device: str) -> None:
    if _state["zimage_pipe"] is not None and _state["zimage_device"] == device:
        return
    zimage_pipe = ZImagePipeline.from_pretrained(
        ZIMAGE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    zimage_pipe.to(device)
    _state["zimage_pipe"] = zimage_pipe
    _state["zimage_device"] = device


@app.on_event("startup")
def startup_event() -> None:
    _ensure_results_dir()
    _load_qwen_model(f"cuda:{DEFAULT_QWEN_GPU_ID}")
    _load_zimage_model(f"cuda:{DEFAULT_ZIMAGE_GPU_ID}")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "qwen_device": _state["qwen_device"],
        "zimage_device": _state["zimage_device"],
        "qwen_model_path": QWEN_MODEL_PATH,
        "zimage_model_path": ZIMAGE_MODEL_PATH,
        "results_dir": str(RESULTS_DIR),
    }


def _open_image_from_upload_or_path(upload: Optional[UploadFile], image_path: Optional[str]) -> Image.Image:
    if upload is None and not image_path:
        raise HTTPException(status_code=400, detail="Provide either image file upload or image_path")
    if upload is not None and image_path:
        raise HTTPException(status_code=400, detail="Use only one of: upload or image_path")

    if upload is not None:
        raw = upload.file.read()
        return Image.open(io.BytesIO(raw)).convert("RGB")

    assert image_path is not None
    p = Path(image_path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=400, detail=f"image_path not found: {image_path}")
    return Image.open(p).convert("RGB")


def _generate_caption(
    image: Image.Image,
    user_input: str,
    min_pixels: int,
    max_pixels: int,
    max_new_tokens: int,
) -> str:
    processor = _state["processor"]
    qwen_model = _state["qwen_model"]
    qwen_device = _state["qwen_device"]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": user_input},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(qwen_device) if hasattr(v, "to") else v for k, v in inputs.items()}

    generated_ids = qwen_model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


def _run_zimage(
    prompt: str,
    init_image: Image.Image,
    seed: Optional[int],
    negative_prompt: Optional[str],
    guidance_scale: float,
    num_inference_steps: int,
    width: Optional[int],
    height: Optional[int],
) -> Image.Image:
    pipe = _state["zimage_pipe"]
    zimage_device = _state["zimage_device"]

    generator = None
    if seed is not None:
        generator = torch.Generator(device=zimage_device).manual_seed(seed)

    call_sig = inspect.signature(pipe.__call__)
    call_params = set(call_sig.parameters.keys())

    kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
    }

    # 不同版本/实现的 ZImagePipeline 对输入图参数命名不一致
    # 常见命名：image / init_image / image_prompt / input_image
    image_param_candidates = ("image", "init_image", "image_prompt", "input_image")
    image_param = next((name for name in image_param_candidates if name in call_params), None)
    if image_param is None:
        raise RuntimeError(
            f"ZImagePipeline.__call__ does not accept image inputs. signature={call_sig}"
        )
    kwargs[image_param] = init_image
    if negative_prompt and "negative_prompt" in call_params:
        kwargs["negative_prompt"] = negative_prompt
    if width and "width" in call_params:
        kwargs["width"] = width
    if height and "height" in call_params:
        kwargs["height"] = height
    if generator is not None and "generator" in call_params:
        kwargs["generator"] = generator

    out = pipe(**kwargs)
    return out.images[0]


def _save_result(image: Image.Image) -> str:
    filename = f"result1_{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
    output_path = RESULTS_DIR / filename
    image.save(output_path)
    return str(output_path)


def _image_to_base64(image: Image.Image) -> str:
    buff = io.BytesIO()
    image.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


@app.post("/infer/result1")
def infer_result1(
    image: Optional[UploadFile] = File(default=None),
    image_path: Optional[str] = Form(default=None),
    user_input: str = Form(default="Describe this image for re-generation."),
    min_pixels: int = Form(default=117600),
    max_pixels: int = Form(default=786432),
    max_new_tokens: int = Form(default=1024),
    qwen_gpu_id: Optional[int] = Form(default=None),
    zimage_gpu_id: Optional[int] = Form(default=None),
    seed: Optional[int] = Form(default=None),
    prompt_override: Optional[str] = Form(default=None),
    negative_prompt: Optional[str] = Form(default=None),
    guidance_scale: float = Form(default=1.0),
    num_inference_steps: int = Form(default=4),
    width: Optional[int] = Form(default=None),
    height: Optional[int] = Form(default=None),
    return_base64: bool = Form(default=False),
) -> Dict[str, Any]:
    start = time.time()

    target_qwen_device = f"cuda:{DEFAULT_QWEN_GPU_ID if qwen_gpu_id is None else qwen_gpu_id}"
    target_zimage_device = f"cuda:{DEFAULT_ZIMAGE_GPU_ID if zimage_gpu_id is None else zimage_gpu_id}"

    with _infer_lock:
        if _state["qwen_device"] != target_qwen_device:
            _load_qwen_model(target_qwen_device)
        if _state["zimage_device"] != target_zimage_device:
            _load_zimage_model(target_zimage_device)

        src_image = _open_image_from_upload_or_path(image, image_path)
        caption_text = _generate_caption(src_image, user_input, min_pixels, max_pixels, max_new_tokens)
        final_prompt = prompt_override or caption_text

        result_img = _run_zimage(
            prompt=final_prompt,
            init_image=src_image,
            seed=seed,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
        )
        output_path = _save_result(result_img)

    response: Dict[str, Any] = {
        "success": True,
        "caption_text": caption_text,
        "final_prompt": final_prompt,
        "result_image_path": output_path,
        "actual_qwen_device": target_qwen_device,
        "actual_zimage_device": target_zimage_device,
        "latency_ms": int((time.time() - start) * 1000),
        "used_params": {
            "seed": seed,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "width": width,
            "height": height,
            "max_new_tokens": max_new_tokens,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        },
    }
    if return_base64:
        response["result_image_base64"] = _image_to_base64(result_img)
    return response


@app.post("/infer/result1_by_path")
def infer_result1_by_path(req: InferPathRequest) -> Dict[str, Any]:
    target_qwen_device = f"cuda:{DEFAULT_QWEN_GPU_ID if req.qwen_gpu_id is None else req.qwen_gpu_id}"
    target_zimage_device = f"cuda:{DEFAULT_ZIMAGE_GPU_ID if req.zimage_gpu_id is None else req.zimage_gpu_id}"

    with _infer_lock:
        if _state["qwen_device"] != target_qwen_device:
            _load_qwen_model(target_qwen_device)
        if _state["zimage_device"] != target_zimage_device:
            _load_zimage_model(target_zimage_device)

        src_image = _open_image_from_upload_or_path(None, req.image_path)
        caption_text = _generate_caption(src_image, req.user_input, req.min_pixels, req.max_pixels, req.max_new_tokens)
        final_prompt = req.prompt_override or caption_text
        result_img = _run_zimage(
            prompt=final_prompt,
            init_image=src_image,
            seed=req.seed,
            negative_prompt=req.negative_prompt,
            guidance_scale=req.guidance_scale,
            num_inference_steps=req.num_inference_steps,
            width=req.width,
            height=req.height,
        )
        output_path = _save_result(result_img)

    return {
        "success": True,
        "caption_text": caption_text,
        "final_prompt": final_prompt,
        "result_image_path": output_path,
        "actual_qwen_device": target_qwen_device,
        "actual_zimage_device": target_zimage_device,
    }
