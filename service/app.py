import base64
import io
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

# Path conventions requested by user
BASE_DIR = Path(os.getenv("COMFYUI_BASE_DIR", "/data1/w00916456/ComfyUI/ComfyUIDeploy"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", str(BASE_DIR / "results"))).resolve()
QWEN_MODEL_PATH = os.getenv("QWEN_MODEL_PATH", "/data1/w00916456/Qwen3-VL/Qwen3-VL-32B-Instruct")
ZIMAGE_MODEL_PATH = os.getenv("ZIMAGE_MODEL_PATH", "/data1/w00916456/Z-Image-main/Z-Image-Turbo")
DEFAULT_QWEN_GPU_ID = int(os.getenv("DEFAULT_QWEN_GPU_ID", "0"))
DEFAULT_ZIMAGE_GPU_ID = int(os.getenv("DEFAULT_ZIMAGE_GPU_ID", "1"))

app = FastAPI(title="Result1 API", version="1.0.0")

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


def _ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_qwen(device: str) -> None:
    if _state["qwen_model"] is not None and _state["qwen_device"] == device:
        return
    _state["qwen_model"] = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_PATH,
        torch_dtype="auto",
        device_map=device,
    )
    _state["processor"] = AutoProcessor.from_pretrained(QWEN_MODEL_PATH)
    _state["qwen_device"] = device


def _load_zimage(device: str) -> None:
    if _state["zimage_pipe"] is not None and _state["zimage_device"] == device:
        return
    pipe = ZImagePipeline.from_pretrained(
        ZIMAGE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to(device)
    _state["zimage_pipe"] = pipe
    _state["zimage_device"] = device


def _resolve_devices(qwen_gpu_id: Optional[int], zimage_gpu_id: Optional[int]) -> tuple[str, str]:
    qwen = DEFAULT_QWEN_GPU_ID if qwen_gpu_id is None else qwen_gpu_id
    zimg = DEFAULT_ZIMAGE_GPU_ID if zimage_gpu_id is None else zimage_gpu_id
    return f"cuda:{qwen}", f"cuda:{zimg}"


def _prepare_image(upload: Optional[UploadFile], image_path: Optional[str]) -> Image.Image:
    if upload is None and not image_path:
        raise HTTPException(status_code=400, detail="Provide either image upload or image_path")
    if upload is not None and image_path:
        raise HTTPException(status_code=400, detail="Use only one of image/upload or image_path")
    if upload is not None:
        return Image.open(io.BytesIO(upload.file.read())).convert("RGB")
    assert image_path is not None
    p = Path(image_path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=400, detail=f"image_path not found: {image_path}")
    return Image.open(p).convert("RGB")


def _caption(image: Image.Image, user_input: str, min_pixels: int, max_pixels: int, max_new_tokens: int) -> str:
    processor = _state["processor"]
    model = _state["qwen_model"]
    device = _state["qwen_device"]

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image, "min_pixels": min_pixels, "max_pixels": max_pixels},
        {"type": "text", "text": user_input},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def _generate(
    image: Image.Image,
    prompt: str,
    seed: Optional[int],
    negative_prompt: Optional[str],
    guidance_scale: float,
    num_inference_steps: int,
    width: Optional[int],
    height: Optional[int],
) -> Image.Image:
    pipe = _state["zimage_pipe"]
    generator = None
    if seed is not None:
        generator = torch.Generator(device=_state["zimage_device"]).manual_seed(seed)

    kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "image": image,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
    }
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    if width:
        kwargs["width"] = width
    if height:
        kwargs["height"] = height
    if generator is not None:
        kwargs["generator"] = generator

    return pipe(**kwargs).images[0]


def _save(img: Image.Image) -> str:
    out = RESULTS_DIR / f"result1_{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
    img.save(out)
    return str(out)


def _b64(img: Image.Image) -> str:
    b = io.BytesIO()
    img.save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode("utf-8")


@app.on_event("startup")
def _startup() -> None:
    _ensure_dirs()
    qd, zd = _resolve_devices(None, None)
    _load_qwen(qd)
    _load_zimage(zd)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "qwen_device": _state["qwen_device"],
        "zimage_device": _state["zimage_device"],
        "qwen_model_path": QWEN_MODEL_PATH,
        "zimage_model_path": ZIMAGE_MODEL_PATH,
        "qwen_model_exists": Path(QWEN_MODEL_PATH).exists(),
        "zimage_model_exists": Path(ZIMAGE_MODEL_PATH).exists(),
        "comfyui_base_dir": str(BASE_DIR),
        "comfyui_base_exists": BASE_DIR.exists(),
        "results_dir": str(RESULTS_DIR),
        "results_dir_exists": RESULTS_DIR.exists(),
    }


@app.get("/deploy/check")
def deploy_check() -> Dict[str, Any]:
    return health()


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
    qd, zd = _resolve_devices(qwen_gpu_id, zimage_gpu_id)

    with _infer_lock:
        _load_qwen(qd)
        _load_zimage(zd)
        src = _prepare_image(image, image_path)
        cap = _caption(src, user_input, min_pixels, max_pixels, max_new_tokens)
        prompt = prompt_override or cap
        out_img = _generate(src, prompt, seed, negative_prompt, guidance_scale, num_inference_steps, width, height)
        out_path = _save(out_img)

    resp: Dict[str, Any] = {
        "success": True,
        "caption_text": cap,
        "final_prompt": prompt,
        "result_image_path": out_path,
        "actual_qwen_device": qd,
        "actual_zimage_device": zd,
        "latency_ms": int((time.time() - start) * 1000),
    }
    if return_base64:
        resp["result_image_base64"] = _b64(out_img)
    return resp


@app.post("/infer/result1_by_path")
def infer_result1_by_path(req: InferPathRequest) -> Dict[str, Any]:
    with _infer_lock:
        qd, zd = _resolve_devices(req.qwen_gpu_id, req.zimage_gpu_id)
        _load_qwen(qd)
        _load_zimage(zd)
        src = _prepare_image(None, req.image_path)
        cap = _caption(src, req.user_input, req.min_pixels, req.max_pixels, req.max_new_tokens)
        prompt = req.prompt_override or cap
        out_img = _generate(src, prompt, req.seed, req.negative_prompt, req.guidance_scale, req.num_inference_steps, req.width, req.height)
        out_path = _save(out_img)

    return {
        "success": True,
        "caption_text": cap,
        "final_prompt": prompt,
        "result_image_path": out_path,
        "actual_qwen_device": qd,
        "actual_zimage_device": zd,
    }
