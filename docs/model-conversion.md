# Model Conversion Guide

将 YOLO 权重转换为 TensorRT engine 的完整流程。

## 推荐流程

```
yolov8n.pt  →  yolov8n.onnx (dynamic)  →  yolov8n_dynamic_b16.engine
```

**必须用动态 ONNX**：静态 ONNX（batch 维固定）编出的 engine `getTensorShape` 无 `-1`，但若用旧方法编出含 `-1` 的 engine 却 profile 只覆盖 batch=16，运行时会出现 `Static dimension mismatch` 警告并以错误 shape 推理。

---

## Step 1：导出动态 ONNX

```bash
yolo export \
  model=/models/yolov8n.pt \
  format=onnx \
  dynamic=True \
  batch=16 \
  imgsz=640 \
  half=True
```

输出：`/models/yolov8n.onnx`

---

## Step 2：编译 TensorRT engine

```bash
docker run --rm \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v ./models:/models \
  --entrypoint trtexec \
  registry.cn-hangzhou.aliyuncs.com/daxx/inference-server-tensorrt:main \
    --onnx=/models/yolov8n.onnx \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:8x3x640x640 \
    --maxShapes=images:16x3x640x640 \
    --saveEngine=/models/yolov8n_dynamic_b16.engine \
    --fp16
```

| 参数 | 说明 |
|------|------|
| `--minShapes` | 最小 batch=1，支持单帧推理 |
| `--optShapes` | 优化目标 batch=8，TRT 以此调优 kernel |
| `--maxShapes` | 最大 batch=16，与 `config.yaml` 的 `batch_size` 一致 |
| `--fp16` | 半精度，速度约 2×，精度损失可忽略 |

编译时间约 2–5 分钟，取决于 GPU 型号。

---

## Step 3：更新配置并重启

```yaml
# config.yaml
models:
  - engine_path: /models/yolov8n_dynamic_b16.engine
    batch_size: 16
```

```bash
docker compose restart infer-trt
```

验证警告消失：

```bash
docker compose logs -f infer-trt | grep -E "WARN|ERROR|setInputShape"
```

---

## 不同模型尺寸

| 模型 | pt 文件 | 推荐 optShapes batch |
|------|---------|---------------------|
| YOLOv8n | yolov8n.pt | 8 |
| YOLOv8s | yolov8s.pt | 8 |
| YOLOv8m | yolov8m.pt | 4 |
| YOLOv8l | yolov8l.pt | 4 |
| YOLO11n | yolo11n.pt | 8 |
| YOLO11s | yolo11s.pt | 8 |
| YOLO11m | yolo11m.pt | 4 |
| YOLO26（占位）| — | — |

batch 越大显存占用越高，`optShapes` 建议设为实际平均吞吐量对应的 batch 大小。

> **注意**：YOLO11 后处理格式与 YOLOv8 相同，导出与编译命令一致，将 `model=` 参数替换为对应 `.pt` 即可。YOLO26Decoder 当前为占位实现（返回空结果），格式确认后补全；转换流程与 v8/v11 相同。
