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
RTSP(source) -> decode.ffmpeg -> (fan-out)
            -> archive.raw (可选)
            -> preprocess.yolo -> infer.engine -> postprocess.yolo
            -> track.bytetrack (可选)
            -> join.byFrameId (可选)
            -> sink.kafka
            -> sink.stream (可选，画框+RTSP/RTMP 推流)
            -> sink.ffplay (可选，画框+本机 ffplay 预览)
            -> ManagementServer (/healthz /metrics /tasks)
```

运行时关键点：

- Pipeline 可编排（DAG）：用 `sources` + `pipelines`（图模板）+ `tasks`（`source_id` + `pipeline_id`）声明拓扑与运行实例，支持分支并行与汇合。
- 模型配置仍集中在 `models`，由 `infer.engine` stage 引用（`with.model_id`）。

## 依赖

| 依赖 | 版本要求 | 说明 |
| --- | --- | --- |
| CMake | >= 3.20 | 必需 |
| C++ 编译器 | C++17 | 必需 |
| FFmpeg | 4.x / 5.x / 6.x | 必需 |
| OpenCV | >= 4.0（core,imgproc,imgcodecs,videoio） | 必需 |
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

Kafka 可观测 topic：
- `inference-heartbeat`：引擎与 stream 心跳（含 `stream_state`）
- `inference-control`：stream 控制事件（断流/恢复/终态失败）

### Task 管理

```bash
# 列出 task 与状态
curl http://localhost:8080/tasks

# 启动 / 停止某个 task（id 与配置中 tasks[].id 一致，例如 task_cam_001）
curl -X POST http://localhost:8080/tasks/task_cam_001/start
curl -X POST http://localhost:8080/tasks/task_cam_001/stop
```

## Pipeline 配置（新格式）

配置文件以 `sources` 描述输入源（`id`、`url` 与重连相关字段），以 `pipelines` 描述可编排 DAG 模板（nodes/edges），以 `tasks` 绑定「哪路源跑哪张图」。
每条 `tasks` 可单独设置 **`sample_fps`**（默认 `5`，须 ≥ 1）与 **`use_hwdec`**（默认 `false`）；二者已从 `sources` 迁出，若在 `sources` 下仍写 `sample_fps` / `use_hwdec`，加载配置时会报错提示迁移到对应 task。
示例见 `config/config.cpu.yaml` / `config/config.gpu.yaml` / `config/config.yaml`。

常见 stage（首批）：
- `source.rtsp`：RTSP/文件输入（内部使用 `FFmpegDecoder`）
- `decode.ffmpeg`：解码阶段（当前为占位 passthrough，解码由 source 完成）
- `archive.raw`：原图归档（复用 `FrameArchiver`，支持 `use_hwdec=true` 的 GPU 帧，默认开启）
  - 可通过 `frame_archive.worker_count` 配置归档并发 worker 数（默认 `1`）。
- `preprocess.yolo` / `postprocess.yolo`：占位 passthrough（后续可落地真实算子）
- `infer.engine`：推理 stage（引用 `models[].id`）
- `track.bytetrack`：ByteTrack 追踪
- `join.byFrameId`：归档信息回填到推理结果（按 frame id join）
- `sink.kafka`：Kafka 输出
- `sink.stream`：叠加检测框/标签后推流（支持 `protocol=rtsp|rtmp`，需 `output_url`）
- `sink.ffplay`：叠加检测框后通过管道喂给本机 `ffplay`（BGR rawvideo，需已安装 ffmpeg/ffplay）

### ONNX Runtime（CPU/MPS）注意事项

当前 `OnnxBackend` 默认 **按 batch=1 运行**（一些 ORT 版本在输入 shape 自省上可能不稳定；且多数导出的 YOLO ONNX 为静态 batch=1）。
如果需要 batch>1，请导出支持 batching 的 ONNX 并在后端能力就绪后放开该限制。

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

- 在 pipeline 中通过 `track.bytetrack` stage 启用追踪；`deepsort` 仍为占位。
- `bytetrack` 已实现，会在 Kafka 输出 `detections[].track_id` 可选字段。
- `deepsort` 当前为占位配置，运行时会输出未实现提示并跳过追踪。
- `track.bytetrack.with.*` 支持配置阈值与生命周期参数（同默认值语义）。

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