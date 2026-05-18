# Result1 Inference API (Qwen3-VL-32B-Instruct + Z-Image-Turbo)

该服务用于最小化打通 `结果1` 通路：

`输入图像 -> Qwen3-VL 生成描述 -> Z-Image-Turbo 生成结果图 -> 保存到 ./results`

## 目录约定

- ComfyUI 目录：`/data1/w00916456/ComfyUI`
  - 当前仓库：`/data1/w00916456/ComfyUI/ComfyUIDeploy`
  - 官方仓库：`/data1/w00916456/ComfyUI/ComfyUI-0.21.1`
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
export QWEN_MODEL_PATH=/data1/w00916456/Qwen3-VL/Qwen3-VL-32B-Instruct
export ZIMAGE_MODEL_PATH=/data1/w00916456/Z-Image-main/Z-Image-Turbo
export DEFAULT_QWEN_GPU_ID=0
export DEFAULT_ZIMAGE_GPU_ID=1
uvicorn service.app:app --host 0.0.0.0 --port 38188
```

## 接口

- `GET /health`
- `POST /infer/result1`：支持上传文件 或 `image_path`
- `POST /infer/result1_by_path`：JSON 方式按路径传图

### 关键参数

- `qwen_gpu_id`: Qwen3-VL 使用的 GPU
- `zimage_gpu_id`: Z-Image-Turbo 使用的 GPU
- `max_new_tokens`: Qwen 生成长度（默认 1024，可调）
- `seed/guidance_scale/num_inference_steps/width/height/negative_prompt`: Z-Image 参数

### 示例：文件上传

```bash
curl -X POST 'http://127.0.0.1:38188/infer/result1' \
  -F 'image=@/data1/w00916456/test.png' \
  -F 'user_input=请详细描述图像风格并生成重绘提示词' \
  -F 'qwen_gpu_id=0' \
  -F 'zimage_gpu_id=1' \
  -F 'seed=42' \
  -F 'guidance_scale=1.0' \
  -F 'num_inference_steps=4' \
  -F 'max_new_tokens=1024'
```

### 示例：路径输入

```bash
curl -X POST 'http://127.0.0.1:38188/infer/result1' \
  -F 'image_path=/data1/w00916456/test.png' \
  -F 'qwen_gpu_id=0' -F 'zimage_gpu_id=1'
```

## ComfyUI 快速对接（API 被 ComfyUI 调用）

> 目标：在 ComfyUI Web UI 中一键调用 `/infer/result1`。

### 方案A（推荐，最快）：HTTP Request 节点

1. 在 ComfyUI 安装支持 HTTP POST 的自定义节点（常见 HTTP Request/REST 节点均可）。
2. 节点配置：
   - Method: `POST`
   - URL: `http://127.0.0.1:38188/infer/result1`（若 API 与 ComfyUI 在同容器）
   - Content-Type: `multipart/form-data`
3. 表单字段映射：
   - `image` <- ComfyUI 的 LoadImage 输出文件
   - `user_input` <- 文本输入节点
   - `qwen_gpu_id` <- 整数参数节点（如 0）
   - `zimage_gpu_id` <- 整数参数节点（如 1）
   - `max_new_tokens` / `seed` / `guidance_scale` / `num_inference_steps` 等 <- 参数节点
4. 解析返回 JSON 的 `result_image_path`，再用 LoadImage(From Path) 节点加载结果图显示。

### 方案B（无需安装节点）：在 ComfyUI 的 Python 自定义节点里直接 requests.post

- 在自定义节点中将 ComfyUI 输入图保存到临时文件；
- 调用 `POST /infer/result1` 传 `image_path` 与参数；
- 从返回的 `result_image_path` 读图回传给下游节点。

## 说明

- 模型在服务启动时加载一次，不会每请求重复 `from_pretrained`。
- Qwen 与 Z-Image 可在不同 GPU 上运行（分别由 `qwen_gpu_id` 和 `zimage_gpu_id` 指定）。
- 若请求时切换到新 GPU，会触发对应模型重载（耗时较高）；生产建议固定 GPU 实例化服务。
