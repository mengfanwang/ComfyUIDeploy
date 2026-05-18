# Result1 Inference API (Qwen3-VL-32B-Instruct + Z-Image-Turbo)

本 README 提供**从 0 开始**的完整部署步骤，适用于：

- 你只有自己的基础镜像（例如 `knowbase:0.0.4`）
- 还没有 ComfyUI
- 还没有 Python 推理接口
- 还没有 HTTP 服务

目标是打通：

`输入图像 -> Qwen3-VL 生成描述 -> Z-Image-Turbo 生成图 -> 保存到 /data1/w00916456/ComfyUI/ComfyUIDeploy/results`

---

## 0. 前置假设

- 服务器可用 NVIDIA GPU（如 A100）
- 宿主机已安装可用的 NVIDIA Driver + Docker + nvidia-container-runtime
- 模型已在宿主机目录：
  - `/data1/w00916456/Qwen3-VL/Qwen3-VL-32B-Instruct`
  - `/data1/w00916456/Z-Image-main/Z-Image-Turbo`

---

## 1. 启动你的基础容器

```bash
sudo docker run -it --name knowbase_result1 \
  --gpus all \
  -p 39188:39188 \
  -p 38188:38188 \
  -v /data1/w00916456:/data1/w00916456 \
  knowbase:0.0.4
```

说明：
- `39188` 用于 ComfyUI Web UI。
- `38188` 用于本项目推理 API。

---

## 2. 在容器内准备目录

```bash
mkdir -p /data1/w00916456/ComfyUI
mkdir -p /data1/w00916456/ComfyUI/ComfyUIDeploy/results
```

---

## 3. 从官方仓库部署 ComfyUI（从 0 开始必须做）

```bash
cd /data1/w00916456/ComfyUI
# 如首次部署到固定目录 ComfyUI-0.21.1
git clone https://github.com/comfyanonymous/ComfyUI.git ComfyUI-0.21.1
# 若目录已存在则更新
# cd ComfyUI-0.21.1 && git pull
```

安装依赖：

```bash
cd /data1/w00916456/ComfyUI/ComfyUI-0.21.1
pip install -r requirements.txt
```

启动 ComfyUI：

```bash
cd /data1/w00916456/ComfyUI/ComfyUI-0.21.1
python main.py --listen 0.0.0.0 --port 39188
```

访问：`http://<服务器IP>:39188`

---

## 4. 部署本仓库的 Python 推理 API

> 假设本仓库代码在容器内可见路径 `/data1/w00916456/ComfyUI/ComfyUIDeploy`。

安装依赖：

```bash
cd /data1/w00916456/ComfyUI/ComfyUIDeploy
pip install -r service/requirements.txt
```

设置环境变量并启动：

```bash
cd /data1/w00916456/ComfyUI/ComfyUIDeploy
export COMFYUI_BASE_DIR=/data1/w00916456/ComfyUI/ComfyUIDeploy
export RESULTS_DIR=/data1/w00916456/ComfyUI/ComfyUIDeploy/results
export QWEN_MODEL_PATH=/data1/w00916456/Qwen3-VL/Qwen3-VL-32B-Instruct
export ZIMAGE_MODEL_PATH=/data1/w00916456/Z-Image-main/Z-Image-Turbo
export DEFAULT_QWEN_GPU_ID=0
export DEFAULT_ZIMAGE_GPU_ID=1
uvicorn service.app:app --host 0.0.0.0 --port 38188
```

健康检查：

```bash
curl http://127.0.0.1:38188/health
```

---

## 5. 验证 API（先不经过 ComfyUI）

### 5.1 用 `image_path` 调用（推荐先测）

```bash
curl -X POST 'http://127.0.0.1:38188/infer/result1' \
  -F 'image_path=/data1/w00916456/test.png' \
  -F 'user_input=请详细描述图像风格并生成重绘提示词' \
  -F 'qwen_gpu_id=0' \
  -F 'zimage_gpu_id=1' \
  -F 'max_new_tokens=1024' \
  -F 'seed=42' \
  -F 'guidance_scale=1.0' \
  -F 'num_inference_steps=4'
```

### 5.2 文件上传调用

```bash
curl -X POST 'http://127.0.0.1:38188/infer/result1' \
  -F 'image=@/data1/w00916456/test.png' \
  -F 'qwen_gpu_id=0' \
  -F 'zimage_gpu_id=1'
```

成功后会在 `/data1/w00916456/ComfyUI/ComfyUIDeploy/results` 看到结果图。

---

## 6. 在 ComfyUI 安装 HTTP POST 自定义节点

在 ComfyUI 目录执行（示例流程）：

```bash
cd /data1/w00916456/ComfyUI/ComfyUI-0.21.1/custom_nodes
# 选择并安装一个支持 HTTP/REST 的节点仓库
# git clone <某个HTTP节点仓库地址>
```

安装该节点依赖（若仓库提供 requirements）：

```bash
# cd <该节点目录>
# pip install -r requirements.txt
```

重启 ComfyUI。

> 不同 HTTP 节点仓库名字不同，但你需要的能力是：
> - 支持 POST
> - 支持 multipart/form-data
> - 支持解析 JSON 响应

---

## 7. 在 ComfyUI 中配置调用 `/infer/result1`（image_path 方式）

HTTP 节点参数：

- Method: `POST`
- URL: `http://127.0.0.1:38188/infer/result1`
- Content-Type: `multipart/form-data`
- Form fields:
  - `image_path`: `/data1/w00916456/test.png`（或上游节点拼出的路径）
  - `user_input`: 文本提示
  - `qwen_gpu_id`: `0`
  - `zimage_gpu_id`: `1`
  - `max_new_tokens`: `1024`
  - `seed`: `42`
  - `guidance_scale`: `1.0`
  - `num_inference_steps`: `4`

解析 JSON 返回中的 `result_image_path`，再接“按路径读图节点”进行展示。

---

## 8. 接口说明

- `GET /health`
- `POST /infer/result1`：支持上传文件或 `image_path`（二选一）
- `POST /infer/result1_by_path`：JSON 方式按路径传图

关键参数：

- `qwen_gpu_id`: Qwen3-VL 使用 GPU
- `zimage_gpu_id`: Z-Image-Turbo 使用 GPU
- `max_new_tokens`: Qwen 输出长度（可调）
- `seed/guidance_scale/num_inference_steps/width/height/negative_prompt`: Z-Image 参数

---

## 9. 常见问题

1. **为什么第一次慢？**
   - 模型首次加载很重，尤其 Qwen3-VL-32B。
2. **为什么切换 GPU 会更慢？**
   - 切换 `qwen_gpu_id` 或 `zimage_gpu_id` 会触发该模型重载。
3. **是否每次请求都加载模型？**
   - 不会，服务启动后常驻内存；仅在切 GPU 时重载。

