# Result1 Inference API (Qwen3-VL-32B-Instruct + Z-Image-Turbo)

该服务用于最小化打通 `结果1` 通路：

`输入图像 -> Qwen3-VL 生成描述 -> Z-Image-Turbo 生成结果图 -> 保存到 ./results`

## 目录约定

- ComfyUI 根目录：`/data1/w00916456/ComfyUI`
- 输出目录：`/data1/w00916456/ComfyUI/results`

## 安装

```bash
pip install -r service/requirements.txt
```

## 启动

推荐端口：`38188`（避开常见 8188 冲突）

```bash
export COMFYUI_BASE_DIR=/data1/w00916456/ComfyUI
export RESULTS_DIR=/data1/w00916456/ComfyUI/results
export QWEN_MODEL_PATH=/data1/w00916456/Qwen3-VL-32B-Instruct
export ZIMAGE_MODEL_PATH=/data1/w00916456/Z-Image-main/Z-Image-Turbo
export DEFAULT_GPU_ID=0
uvicorn service.app:app --host 0.0.0.0 --port 38188
```

## 接口

- `GET /health`
- `POST /infer/result1`：支持上传文件 或 `image_path`
- `POST /infer/result1_by_path`：JSON 方式按路径传图

### 示例：文件上传

```bash
curl -X POST 'http://127.0.0.1:38188/infer/result1' \
  -F 'image=@/data1/w00916456/test.png' \
  -F 'user_input=请详细描述图像风格并生成重绘提示词' \
  -F 'seed=42' \
  -F 'guidance_scale=1.0' \
  -F 'num_inference_steps=4' \
  -F 'max_new_tokens=1024' \
  -F 'gpu_id=0'
```

### 示例：路径输入

```bash
curl -X POST 'http://127.0.0.1:38188/infer/result1' \
  -F 'image_path=/data1/w00916456/test.png' \
  -F 'seed=42' -F 'gpu_id=0'
```

## 与 ComfyUI 对接

- 在 ComfyUI 中使用 HTTP Request 类节点（或自定义 Python 节点）调用 `/infer/result1`。
- 第一阶段建议单并发（单请求串行），稳定后再调高。

## 说明

- 模型在服务启动时加载一次，不会每请求重复 `from_pretrained`。
- 若传入不同 `gpu_id`，服务会切换设备并重载模型（耗时较高）。建议生产中固定单卡运行一个服务实例。
