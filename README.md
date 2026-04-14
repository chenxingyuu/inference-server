# inference-server

C++ 实时视频推理服务，支持 100 路 RTSP 摄像头并发接入，目标端到端延迟 < 100ms。

## 架构概览

```
RTSP 摄像头
    │
    ▼
FFmpegDecoder          软解: CPU → cv::Mat → H2D memcpy
(NVDEC 可选)           硬解: NVDEC → NV12 GPU buffer → 零拷贝
    │
    ▼
FrameBuffer            boost::lockfree::spsc_queue<32>，无锁
    │
    ▼
BatchScheduler         按 model_id 筛流，max_batch=16，max_wait=10ms
    │
    ▼
InferWorker
    ├─ TRTBackend      TensorRT：软解走 CPU 预处理；硬解走 fused CUDA kernel
    └─ AscendBackend   Ascend 310P：预编译 batch=1/4/8/16 的 .om，运行时选最近档
    │
    ▼
IYOLODecoder           YOLOv5 / v8 / v11 / v26 后处理 + NMS
    │
    ▼
KafkaPublisher         librdkafka 异步生产，JSON 格式
    │
    ▼
ManagementServer       HTTP /healthz /metrics /streams (热增删流)
```

## 快速开始

### 依赖


| 依赖            | 版本              | 必需          |
| ------------- | --------------- | ----------- |
| CMake         | ≥ 3.20          | ✓           |
| C++ 编译器       | C++17           | ✓           |
| FFmpeg        | 4.x / 5.x / 6.x | ✓           |
| OpenCV        | ≥ 4.0           | ✓           |
| yaml-cpp      | any             | ✓           |
| nlohmann/json | ≥ 3.9           | ✓           |
| spdlog        | any             | ✓           |
| Boost         | ≥ 1.71          | ✓           |
| librdkafka    | any             | ✓           |
| TensorRT      | ≥ 8.5           | TRT only    |
| CUDA Toolkit  | ≥ 11.8          | TRT only    |
| Ascend CANN   | 6.x             | Ascend only |


### 编译

```bash
# TensorRT 机器（NVIDIA GPU）
cmake -B build \
  -DBUILD_TRT_BACKEND=ON \
  -DBUILD_ASCEND_BACKEND=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Ascend 310P 机器
cmake -B build \
  -DBUILD_TRT_BACKEND=OFF \
  -DBUILD_ASCEND_BACKEND=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Docker 构建（Ascend）
# - DOCKER_BUILDKIT=1: 启用 BuildKit，才能使用 Dockerfile 中的 apt/ccache 缓存挂载
# - ASCEND_BASE_IMAGE: 可替换 Ascend 基础镜像版本（无需改 Dockerfile）
# - APT_MIRROR: 可选，替换 apt 源以加速依赖下载（Ascend arm64 建议 ubuntu-ports 镜像）
DOCKER_BUILDKIT=1 docker build \
  -t registry.cn-hangzhou.aliyuncs.com/daxx/inference-server:ascend \
  -f docker/Dockerfile.ascend \
  --build-arg ASCEND_BASE_IMAGE=ascendai/cann:8.5.1-310p-ubuntu22.04-py3.11 \
  .

# Ascend 构建（可选：使用国内镜像源加速）
DOCKER_BUILDKIT=1 docker build \
  -t registry.cn-hangzhou.aliyuncs.com/daxx/inference-server:ascend \
  -f docker/Dockerfile.ascend \
  --build-arg ASCEND_BASE_IMAGE=ascendai/cann:8.5.1-310p-ubuntu22.04-py3.11 \
  --build-arg APT_MIRROR=http://mirrors.ustc.edu.cn/ubuntu-ports \
  .

# Docker 构建（TensorRT）
# - CUDA_DEVEL_IMAGE: 编译阶段镜像（含编译工具链）
# - CUDA_RUNTIME_IMAGE: 运行阶段镜像（尽量精简）
# - APT_MIRROR: 可选，替换 apt 源以加速依赖下载（x86 常用 ubuntu 镜像）
# - 首次构建会下载依赖；后续改代码重建会复用缓存，明显更快
DOCKER_BUILDKIT=1 docker build \
  -t registry.cn-hangzhou.aliyuncs.com/daxx/inference-server:tensorrt \
  -f docker/Dockerfile.tensorrt \
  --build-arg CUDA_DEVEL_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 \
  --build-arg CUDA_RUNTIME_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
  .

# TensorRT 构建（可选：使用国内镜像源加速）
DOCKER_BUILDKIT=1 docker build \
  -t registry.cn-hangzhou.aliyuncs.com/daxx/inference-server:tensorrt \
  -f docker/Dockerfile.tensorrt \
  --build-arg CUDA_DEVEL_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 \
  --build-arg CUDA_RUNTIME_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
  --build-arg APT_MIRROR=http://mirrors.ustc.edu.cn/ubuntu \
  .

```

### Docker 启动（TensorRT）

```bash
# 将 .engine 文件放到 models/，修改 config/config.yaml
docker compose -f docker/docker-compose.nvidia.yml up -d
```

启动后包含：

- `infer-trt` — 推理服务，监听 HTTP 8080
- `kafka` — 消息队列
- `prometheus` — 指标采集（9090）
- `grafana` — 可视化（3000，默认密码 `admin`）

### 配置文件

```yaml
# config/config.yaml（关键字段）
server:
  max_streams: 100
  management_port: 8080

models:
  - id: "yolov8n_trt"
    version: yolov8
    backend: tensorrt
    engine_path: "/models/yolov8n_b16.engine"
    batch_size: 16

streams:
  - id: "cam_001"
    url: "rtsp://192.168.1.100/stream1"
    model_id: "yolov8n_trt"
    sample_fps: 5
    use_hwdec: true    # NVDEC 硬解（需 NVIDIA GPU）
```

## HTTP 管理接口

服务启动后在 `management_port`（默认 8080）暴露以下端点：

```bash
# 健康检查
curl http://localhost:8080/healthz

# Prometheus 指标
curl http://localhost:8080/metrics

# 查看当前所有流
curl http://localhost:8080/streams

# 热添加流（无需重启）
curl -X POST http://localhost:8080/streams \
  -H 'Content-Type: application/json' \
  -d '{"id":"cam_003","url":"rtsp://...","model_id":"yolov8n_trt"}'

# 热删除流
curl -X DELETE http://localhost:8080/streams/cam_003

# 优雅关闭
kill -SIGTERM $(pgrep infer_server)
```

## 指标列表


| 指标名                     | 类型        | 标签          | 说明               |
| ----------------------- | --------- | ----------- | ---------------- |
| `infer_latency_ms`      | histogram | `model_id`  | 单批推理耗时           |
| `e2e_latency_ms`        | histogram | `stream_id` | 采集到发布的端到端延迟      |
| `frames_decoded_total`  | counter   | `stream_id` | 已解码帧数            |
| `frames_dropped_total`  | counter   | `stream_id` | 因队列满丢弃的帧数        |
| `kafka_published_total` | counter   | —           | 成功发布到 Kafka 的消息数 |
| `kafka_dropped_total`   | counter   | —           | 因队列满丢弃的消息数       |
| `infer_batches_total`   | counter   | `model_id`  | 已处理批次数           |


## Ascend 310P 模型转换

```bash
# ONNX → .om（batch=1/4/8/16）
bash scripts/convert_ascend.sh yolo11s.onnx yolo11s
```

生成 `yolo11s_b1.om` / `_b4.om` / `_b8.om` / `_b16.om`，在 `config.yaml` 的 `om_paths` 中引用。

## 项目结构

```
inference-server/
├── include/
│   ├── common/        Types.h / Config.h / Logger.h
│   ├── cuda/          CudaPreprocess.h
│   ├── stream/        IStreamDecoder / FFmpegDecoder / FrameBuffer / StreamPool
│   ├── infer/         IInferBackend / TRTBackend / AscendBackend / BackendFactory
│   ├── decoder/       IYOLODecoder / YOLOv5..v26 / DecoderFactory
│   ├── pipeline/      BatchScheduler / InferWorker
│   ├── publisher/     IPublisher / KafkaPublisher
│   ├── metrics/       Metrics.h
│   └── server/        ManagementServer.h
├── src/               对应实现 + src/cuda/CudaPreprocess.cu
├── config/            config.yaml / prometheus.yml / grafana provisioning
├── docker/            Dockerfile.tensorrt / Dockerfile.ascend / docker-compose.*
├── cmake/             FindTensorRT / FindAscendCL / CompilerOptions
├── scripts/           convert_ascend.sh
└── docs/              迭代记录
```

## 开发迭代

详见 [docs/iterations.md](docs/iterations.md)。