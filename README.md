# inference-server

`inference-server` 是一个 C++ 实时视频推理服务，面向多路 RTSP 输入，支持 TensorRT、Ascend、ONNX Runtime（CPU/MPS）三种推理后端、模型热管理（load/unload/swap）、级联分类和 Prometheus 指标暴露。

## 文档导航

- 总入口：[`docs/INDEX.md`](docs/INDEX.md)
- 架构边界：[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- 产品约束：[`docs/PRODUCT_RULES.md`](docs/PRODUCT_RULES.md)
- 质量门禁：[`docs/QUALITY-GATES.md`](docs/QUALITY-GATES.md)
- 迭代记录：[`docs/iterations.md`](docs/iterations.md)

## 架构概览

```text
RTSP -> FFmpegDecoder(可选HW解码) -> FrameBuffer
     -> BatchScheduler(按 model_id 聚合)
     -> InferWorkerGroup
        -> TRTBackend / AscendBackend
     -> YOLO/Classifier Decoder
     -> (可选) CascadeRouter + ResultMerger
     -> KafkaPublisher / AttributePublisher
     -> ManagementServer (/healthz /metrics /streams /models)
```

运行时关键点：

- 流管理与模型管理解耦：流可热增删，模型可热加载/热替换。
- `ModelManager` 统一维护模型状态（`unloaded/loading/ready/draining/unloading`）。
- 支持 detector -> classifier 级联，二级模型由主模型内部托管，不挂到 `streams.model_id`。

## 依赖

| 依赖 | 版本要求 | 说明 |
| --- | --- | --- |
| CMake | >= 3.20 | 必需 |
| C++ 编译器 | C++17 | 必需 |
| FFmpeg | 4.x / 5.x / 6.x | 必需 |
| OpenCV | >= 4.0（core,imgproc） | 必需 |
| yaml-cpp | 任意 | 必需 |
| nlohmann/json | >= 3.9 | 必需 |
| spdlog | 任意 | 必需 |
| Boost | >= 1.71 | 必需 |
| librdkafka | 任意 | 必需 |
| TensorRT | >= 8.5 | TensorRT 后端必需 |
| CUDA Toolkit | >= 11.8 | TensorRT 后端必需 |
| Ascend CANN | 6.x/8.x（按镜像） | Ascend 后端必需 |
| ONNX Runtime | >= 1.18（自动下载） | CPU / MPS 后端必需 |


## Makefile 快速命令

项目根目录提供了统一的 `Makefile`，默认后端为 CPU（ONNX）。

```bash
# 查看所有常用目标
make help

# 本地构建（默认 CPU）
make build

# 指定后端构建
make build-cpu
make build-gpu
make build-npu
make build-tests

# 本地运行（默认使用 config/config.cpu.yaml）
make run

# 覆盖配置文件运行
make run CONFIG=config/config.cpu.yaml

# 质量门禁与测试
make validate
make test
make clean

# Docker Compose 启停
make up
make down
make up-cpu
make down-cpu

# Docker 镜像构建
make docker-build-cpu
make docker-build-gpu
make docker-build-npu
```

## HTTP 管理接口

默认端口来自 `server.management_port`（默认 `8080`）。

### 健康和可观测

```bash
curl http://localhost:8080/healthz
curl http://localhost:8080/metrics
```

### 流管理

```bash
# 列出流
curl http://localhost:8080/streams

# 添加流
curl -X POST http://localhost:8080/streams \
  -H "Content-Type: application/json" \
  -d '{"id":"cam_003","url":"rtsp://...","model_id":"yolov8n_trt","tracker":"bytetrack"}'

# 删除流
curl -X DELETE http://localhost:8080/streams/cam_003
```

### 模型管理

```bash
# 列出模型与状态
curl http://localhost:8080/models

# 加载模型
curl -X POST http://localhost:8080/models \
  -H "Content-Type: application/json" \
  -d '{"id":"yolov8n_trt","backend":"tensorrt","version":"yolov8","engine_path":"/models/yolov8n_b16.engine","batch_size":16}'

# 热替换模型
curl -X PUT http://localhost:8080/models/yolov8n_trt \
  -H "Content-Type: application/json" \
  -d '{"backend":"tensorrt","version":"yolov8","engine_path":"/models/yolov8n_b16_v2.engine","batch_size":16}'

# 卸载模型
curl -X DELETE http://localhost:8080/models/yolov8n_trt

# 查询模型统计
curl http://localhost:8080/models/yolov8n_trt/stats
```

## 核心指标

| 指标名 | 类型 | 标签 | 说明 |
| --- | --- | --- | --- |
| `infer_latency_ms` | histogram | `model_id` | 单批推理耗时 |
| `e2e_latency_ms` | histogram | `stream_id` | 端到端延迟 |
| `frames_decoded_total` | counter | `stream_id` | 解码帧数 |
| `frames_dropped_total` | counter | `stream_id` | 解码丢帧数 |
| `kafka_published_total` | counter | 无 | Kafka 成功发布数 |
| `kafka_dropped_total` | counter | 无 | Kafka 丢弃消息数 |
| `infer_batches_total` | counter | `model_id` | 已处理批次数 |
| `frames_archived_total` | counter | 无 | 本地归档成功帧数 |
| `frames_archive_dropped_total` | counter | 无 | 归档队列溢出或写盘失败 |
| `frames_uploaded_total` | counter | 无 | MinIO 上传成功数 |
| `frames_upload_failed_total` | counter | 无 | MinIO 上传失败数 |
| `frame_archive_queue_depth` | gauge | 无 | 当前归档队列深度 |

## 目标追踪（可选）

- 在 `streams[].tracker` 配置追踪方式：`none`（默认）/ `bytetrack` / `deepsort`。
- `bytetrack` 已实现，会在 Kafka 输出 `detections[].track_id` 可选字段。
- `deepsort` 当前为占位配置，运行时会输出未实现提示并跳过追踪。

## Ascend 模型转换

```bash
# ONNX -> Ascend .om（自动导出 batch=1/4/8/16）
bash scripts/convert_ascend.sh yolo11s.onnx yolo11s
```

转换后可在 `models/` 得到 `*_b1.om / *_b4.om / *_b8.om / *_b16.om`，并在 `config/config.yaml` 的 `om_paths` 中配置。

## 项目结构

```text
inference-server/
├── include/
│   ├── common/       配置/日志/类型
│   ├── stream/       解码与流池
│   ├── infer/        后端与缓冲池
│   ├── decoder/      YOLO 与分类器后处理
│   ├── pipeline/     调度、worker、级联、模型管理
│   ├── tracker/      可选目标追踪（ByteTrack/DeepSORT 占位）
│   ├── publisher/    Kafka 与属性发布
│   ├── metrics/      指标导出
│   ├── server/       HTTP 管理接口
│   ├── archive/      帧归档与 MinIO 上传
│   └── cuda/         CUDA 预处理头文件
├── src/              对应实现
├── config/           服务、Prometheus、Grafana 配置
├── docker/           Dockerfile 与 docker-compose
├── cmake/            CMake 模块
├── scripts/          脚本（convert_ascend.sh / validate-repo.sh）
└── docs/             架构、规则、质量门禁、迭代记录
```

## 开发与提交流程

- 修改代码后同步更新相关文档（`README.md` 或 `docs/**`）。
- 提交前建议运行：

```bash
scripts/validate-repo.sh
```