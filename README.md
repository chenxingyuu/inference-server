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
RTSP(source.rtsp | source.file，内含 FFmpeg/DVPP 解码)
            -> infer.engine（内含 preprocess + YOLO decode）
            -> track.bytetrack (可选)
            -> archive.raw (可选，须在 infer 下游，回填 frame_local_path / frame_url)
            -> sink.publish（publishers: 配置可同时扇出到 Kafka / gRPC / Redis）
            -> sink.stream (可选，画框+RTSP/RTMP 推流)
            -> sink.ffplay (可选，画框+本机 ffplay 预览)

管理面: inferenced (C++) --Unix socket--> infer-server (Go HTTP: /healthz /metrics /tasks ...)
        infer-ctl (Go CLI) 直连同一 socket
```

运行时关键点：

- Pipeline 可编排（DAG）：用 `sources` + `pipelines`（图模板）+ `tasks`（`source_id` + `pipeline_id`）声明拓扑与运行实例，支持分支并行与汇合。
- 模型清单由根配置里的 `models:` 与可选的 **model repository**（`server.model_repository`）在启动时合并；`infer.engine` 通过 `with.model_id` 引用合并后的 `models[].id`。

### Model repository（可选，类 Triton 目录布局）

当 `server.model_repository` 非空（或环境变量 `INFER_MODEL_REPOSITORY` 非空，**环境变量覆盖 YAML**）时，服务在 `loadConfig` 阶段扫描该目录，并把发现的模型 **追加** 到 `AppConfig::models`。与根配置 `models:` 出现 **相同 `id` 会报错**。

目录约定：

```text
<repository>/
  <model_id>/
    config.yaml          # 与根配置里单条 models[] 同字段（不含顶层 models 键）
    <version>/           # 仅正整数字符串目录名，例如 1、2
      *.onnx | *.engine | *.om | ...
```

- **`<model_id>`** 必须匹配 `^[A-Za-z0-9_-]+$`，且与 `config.yaml` 里的 `id`（若填写）一致。
- **`active_version`**（可选，整数）：选用该版本子目录；未设置则取 **最大** 版本号。
- **`weight_file`**（可选，字符串）：所选版本目录内的主权重文件名；未设置则按后端自动探测：TensorRT 取唯一 `.engine`，ONNX 取唯一 `.onnx`，Ascend 在 `om_paths` 未配置时取唯一 `.om`（多文件歧义则报错）。
- `engine_path` / `onnx_path` / `om_paths` 若为相对路径，则相对于 **所选版本目录** 解析。

**限制**：合并发生在进程启动、`TaskManager::loadAll()` 构建 DAG **之前**；DAG 内的 `InferEngineWorkerStage` 在创建时 **拷贝** `ModelConfig`，因此仅改磁盘上的仓库 **不会** 自动更新已运行任务中的权重；需重启进程或重建任务图（与 Triton 在线切版本不同，除非后续做动态注册表）。

### Web 发布模型

在部署环境中设置 `INFER_API_KEY`，然后在 Web 侧边栏填写相同的 API Key。密钥缺失时，上传分析和发布暂停，服务可以照常启动。

在 **模型管理 → 上传模型** 中选择 `.onnx`、`.engine`、`.om` 或单模型 `.zip` / `.tar.gz` / `.tgz`。Ascend 裸文件使用主 `.om` 作为 Batch 1，可在高级设置补充 Batch 4、8、16。压缩包可以包含一份 `config.yaml` 和一组权重，文件可平铺，或使用上面的单版本仓库布局。已有配置会预填参数，发布保留类别名称、输入尺寸与级联设置。Ascend 压缩包中单个权重沿用配置的 `batch_size`。级联二级模型须在根配置或模型仓库中定义并先登记。

核对参数后点击 **校验并发布**。引擎在独立进程里逐个设备试加载权重，成功后将模型登记到仓库的 `<model_id>/1/`。待发布文件保留 30 分钟，引擎不可用或校验失败时可以在同一页面重试。仓库、运行时和当前启动配置中的模型 ID 均参与查重。TAR 压缩包的解压读取总量上限为 4 GiB，包含跳过的条目、归档头部和填充。发布成功后可直接打开流水线编辑器，在 `infer.engine` 中选用新模型。实际推理结果通过启动任务验证。

自行部署时，Go 的 `MODEL_REPOSITORY` 和 C++ 的 `INFER_MODEL_REPOSITORY` 应指向同一可写卷，两个进程的 `INFER_SOCKET` 指向共享 Socket。仓库的 `/models/.uploads` 用于暂存，并由 Go 清理。CPU、NVIDIA 和 Ascend Web Compose 已包含这些设置，部署时使用与本次代码匹配的引擎、管理服务和 Web 镜像。自定义反向代理需允许最多 2 GiB 上传体积和至少 120 秒的发布响应等待。

HTTP 客户端可先向 `POST /models/upload/analyze` 发送 multipart 的 `file` 或 `archive`，再将响应中的 `token`、`id` 和修改过的参数提交给 `POST /models/upload`。两个接口都要求携带 `Authorization: Bearer <INFER_API_KEY>`。引擎未确认发布时，用同一 token 和 ID 重试。旧版直接上传的 multipart 请求继续支持。

### 导出仓库模型

在 **模型管理 → 模型仓库** 点击模型行的 **导出模型**。下载的 `<model_id>-v<活动版本>.zip` 包含当前版本使用的权重和可重新导入的 `config.yaml`。Ascend 包含配置引用的各 Batch 权重，ONNX 包含模型声明的外部张量数据，并保留原有相对路径和文件名。ONNX 依赖必须是模型目录内的普通文件，隐藏路径、符号链接或缺失依赖会在导出时提示错误。上传时直接选择 ZIP，核对预填参数后发布。同一仓库中可将 ID 改为 `原ID-smoke`，再从流水线选择新 ID 跑样例。发布后的仓库版本目录为 `1/`。级联二级模型须先在目标环境登记。

导出和上传使用同一 API Key。导出包遵循上传的 2 GiB 请求限制。大文件可以用命令行直接保存到磁盘，例如在已设置 `INFER_API_KEY` 的终端执行以下命令。管理服务会在临时目录生成 ZIP，自定义代理的读取超时应覆盖生成时间。

```bash
curl --fail --show-error --remote-name --remote-header-name \
  --header "Authorization: Bearer ${INFER_API_KEY}" \
  http://localhost:8080/models/repository/yolov8n_cpu/export
```

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

# 本地运行（默认使用 config/config.yaml）
make run

# 覆盖配置文件运行
make run CONFIG=config/config.yaml

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

## 管理接口

管理接口通过 **Unix Domain Socket** 暴露，路径来自 `server.socket_path`（默认 `/var/run/infer.sock`，可在配置中覆盖）。协议为 newline-delimited JSON，每条连接发一条请求收一条响应。

```bash
# 健康检查
echo '{"cmd":"health"}' | socat - UNIX-CONNECT:./infer.sock

# Prometheus 格式指标
echo '{"cmd":"metrics"}' | socat - UNIX-CONNECT:./infer.sock

# 列出 task 与状态
echo '{"cmd":"list_tasks"}' | socat - UNIX-CONNECT:./infer.sock

# 启动 / 停止某个 task
echo '{"cmd":"start_task","id":"task_cam_001"}' | socat - UNIX-CONNECT:./infer.sock
echo '{"cmd":"stop_task","id":"task_cam_001"}' | socat - UNIX-CONNECT:./infer.sock
```


## Pipeline 配置（新格式）

配置文件以 `sources` 描述输入源（`id`、`url` 与重连相关字段），以 `pipelines` 描述可编排 DAG 模板（nodes/edges），以 `tasks` 绑定「哪路源跑哪张图」。
每条 `tasks` 可单独设置 **`sample_fps`**（默认 `5`，须 ≥ 1）与 **`use_hwdec`**（默认 `false`）；二者已从 `sources` 迁出，若在 `sources` 下仍写 `sample_fps` / `use_hwdec`，加载配置时会报错提示迁移到对应 task。
示例见 `config/config.yaml`。

常见 stage：
- `source.rtsp`：RTSP 输入（内部使用 `FFmpegDecoder`；Ascend 可选 `use_ascend_dvpp` 走 DVPP 硬解）
- `source.file`：本地视频文件输入
- `archive.raw`：原图归档（复用 `FrameArchiver`，支持 `use_hwdec=true` 的 GPU 帧，默认开启）。须放在 `infer.engine` 下游，将 `frame_local_path`（本机绝对路径）与 `frame_url`（完整可访问 URL）写回 `InferResult` 后由 `sink.publish` 发出。
  - 可通过 `frame_archive.worker_count` 配置归档并发 worker 数（默认 `1`）。
  - 本地路径按 UTC 时间分桶：`{YYYYMMDD}/{HH}/{mm}/{stream_id}/{ts_ms}_{seq}.jpg`（见 `include/archive/FrameLayout.h`）。
  - 完整 URL：配置 `frame_archive.public_base_url`（如 `http://frame-nginx:8082/frames`）后，`frame_url = {public_base_url}/{object_key}`，下游直接 `GET` 即可，无需拼前缀。未配置时 `frame_url` 回退为相对 object key（向后兼容，下游需自行拼 `http://<host>:8082/frames/{frame_url}`）。
  - 主进程独立后台组件 `FrameRetentionGc` 按分钟粒度 TTL 整目录删除过期桶（与 `frame_archive.enabled` 解耦；见下方 `retention` 配置）。
- `infer.engine`：推理 stage（引用 `models[].id`）；DAG 路径下由 `InferWorkerGroup` 执行，内含 preprocess 与 YOLO decode，支持 `models[].instance_count` 与 `models[].device_ids`（每实例一个后端；`device_ids` 拼写须正确）。攒批策略与 `batch_size`、`max_queue_delay_us` 一致（与 `ModelManager` + `BatchScheduler` 的流池攒批路径不同）。
- `infer.sahiScheduler`：SAHI 滑窗切块，将大分辨率帧切成重叠 tile 后送入下游 `infer.engine`，最后由 `infer.sahiMerge` 合并 NMS 结果。参数：`tile_width`、`tile_height`、`overlap_ratio`、`full_interval`（每 N 帧插入一次全图推理）、`max_tiles_per_frame`。
- `track.bytetrack`：ByteTrack 追踪
- `sink.publish`：结果输出；通过 `publishers:` 配置可同时扇出到 Kafka / gRPC / Redis（见下方配置示例）
- `sink.stream`：叠加检测框/标签后推流（支持 `protocol=rtsp|rtmp`，需 `output_url`；`encoder` 可选 `ffmpeg_x264`（默认）或 `ascend_venc`）。`output_url` 支持占位符 `{task_id}` / `{source_id}`，在每个 task 构图时插值；未知占位符会报错。
- `sink.ffplay`：叠加检测框后通过管道喂给本机 `ffplay`（BGR rawvideo，需已安装 ffmpeg/ffplay）

`sink.stream` 占位符示例（多 task 复用同一 pipeline 时为每路生成不同推流地址）：

```yaml
pipelines:
  - id: pipe_stream
    nodes:
      - id: sink_stream
        type: sink.stream
        with:
          output_url: "rtsp://push/live/{source_id}/{task_id}"
          protocol: rtsp
```

### 推理节点阈值

在 Web 流水线编辑器中选中 `infer.engine` 节点，通过「推理高级参数」调整置信度和 NMS 阈值。每项独立覆盖该节点主模型的解码阈值，留空时沿用模型配置，取值范围为 `[0, 1]`。YOLO v5、v8、v11 检测模型使用 NMS 阈值，YOLO26 和分类模型只使用置信度阈值。

在 YAML 的节点 `with` 中可以直接设置这两项。HTTP 流水线接口的 `with` 参数仍使用字符串值，例如 `"conf_thresh": "0.3"`。

```yaml
- id: detector
  type: infer.engine
  with:
    model_id: yolov8n_cpu
    conf_thresh: 0.3
    nms_thresh: 0.5
```

更新现有流水线前，先停止使用该流水线的运行中任务。保存会重建相关任务图，任务下次启动时使用新阈值。

### 多路发布配置（publishers:）

`sink.kafka` stage 通过 `buildPublisher()` 工厂自动扇出，多路启用时内部包装为 `MultiPublisher`，单路失败不影响其余路。

```yaml
publishers:
  kafka:
    enabled: true
    # ... 原有 kafka: 根键配置迁移至此（旧格式自动 fallback，零迁移成本）
  grpc:
    enabled: false
    port: 50051
  redis:
    enabled: false
    addr: "localhost:6379"
    stream_key: "infer:results"
```

订阅端示例见 `examples/grpc_subscriber/`（Python / Go）。构建时需加 `-DBUILD_GRPC_PUBLISHER=ON`（需 gRPC）或 `-DBUILD_REDIS_PUBLISHER=ON`（需 hiredis）。

### 帧归档与保留期（`frame_archive`）

```yaml
frame_archive:
  enabled: true
  local_dir: "/data/frames"   # docker compose 中与 infer-frames 卷挂载点一致
  public_base_url: "http://frame-nginx:8082/frames"  # 设置后 frame_url 为完整可访问 URL；留空则回退相对 key
  save_interval: 1          # 每 N 帧写一张
  jpeg_quality: 90
  queue_capacity: 4096
  worker_count: 1
  retention:
    enabled: true
    max_age_minutes: 1440   # TTL，最小粒度 1 分钟
    scan_interval_seconds: 60
```

- **写入**：`FrameArchiver` 异步队列 + 多 worker 写 JPEG，不阻塞推理热路径。
- **发布**：配置 `public_base_url` 后，`frame_url` 为完整可访问 URL（如 `http://frame-nginx:8082/frames/20260424/10/16/cam_01/1777025798000_7.jpg`），Kafka / Redis / gRPC 均携带，下游直接 `GET`；未配置时回退为相对 object key。`frame_local_path` 保留本机绝对路径供同机调试。
- **下载**：docker compose 中的 `frame-nginx`（`:8082`）以只读方式挂载 `infer-frames` 卷，内网直接 `GET frame_url`（完整 URL）或 `GET /frames/{object_key}` 拉 JPEG。
- **回收**：`FrameRetentionGc` 只解析 `YYYYMMDD/HH/mm` 目录名，过期则 `remove_all` 整分钟桶，成本与文件总数无关。
- **部署**：`local_dir` 建议挂独立卷（`infer-frames`）；生产环境务必开启 `retention.enabled`，否则磁盘会无限增长。

### ONNX Runtime（CPU/MPS）注意事项

当前 `OnnxBackend` 默认 **按 batch=1 运行**（一些 ORT 版本在输入 shape 自省上可能不稳定；且多数导出的 YOLO ONNX 为静态 batch=1）。
如果需要 batch>1，请导出支持 batching 的 ONNX 并在后端能力就绪后放开该限制。

## 核心指标

| 指标名 | 类型 | 标签 | 说明 |
| --- | --- | --- | --- |
| `infer_latency_ms` | histogram | `model_id` | 单批推理耗时 |
| `e2e_latency_ms` | histogram | `stream_id` | 端到端延迟 |
| `sink_publish_latency_ms` | histogram | `stream_id` | 帧到达 `sink.publish` 的延迟 |
| `sink_stream_latency_ms` | histogram | `stream_id` | 帧在 `sink.stream` 写流成功时的延迟 |
| `frames_decoded_total` | counter | `stream_id` | 解码帧数 |
| `frames_dropped_total` | counter | `stream_id` | 解码丢帧数 |
| `kafka_published_total` | counter | 无 | Kafka 成功发布数 |
| `kafka_dropped_total` | counter | 无 | Kafka 丢弃消息数 |
| `infer_batches_total` | counter | `model_id` | 已处理批次数 |
| `frames_archived_total` | counter | 无 | 本地归档成功帧数 |
| `frames_archive_dropped_total` | counter | 无 | 归档队列溢出或写盘失败 |
| `frames_archive_deleted_total` | counter | 无 | 保留期 GC 删除的归档条目数 |
| `frame_archive_queue_depth` | gauge | 无 | 当前归档队列深度 |

## 目标追踪（可选）

- 在 pipeline 中通过 `track.bytetrack` stage 启用追踪；`deepsort` 仍为占位。
- `bytetrack` 已实现，会在 Kafka 输出 `detections[].track_id` 可选字段。
- `deepsort` 当前为占位配置，运行时会输出未实现提示并跳过追踪。
- `track.bytetrack.with.*` 支持配置阈值与生命周期参数（同默认值语义）。

## Ascend 模型转换

在仓库根目录执行（脚本会嵌入 `scripts/aipp.cfg`，默认 **640×640** NV12 `src`，无 AIPP resize；1080p 等码流由运行时 **DVPP VPC** 拉伸到模型输入）：

```bash
# ONNX -> Ascend .om（batch=1/4/8/16，含 NV12 AIPP）
./scripts/convert_ascend.sh path/to/yolo11s.onnx yolo11s
```

转换后可在 `models/` 得到 `*_b1.om / *_b4.om / *_b8.om / *_b16.om`，并在 `config/config.yaml` 的 `om_paths` 中配置。

- `scripts/aipp.cfg` 的 `src_image_size_w/h` 必须与 **VPC 输出 / Path B pack 尺寸**一致（通常 640×640），修改后须重新 ATC 并替换 `.om`。
- 任务级 `use_ascend_dvpp: true` 启用 DVPP 硬解；VPC 目标尺寸由 `infer.engine` 对应模型的 `input_shape` 自动注入（见 [docs/ascend-guide.md](docs/ascend-guide.md) §13）。

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
│   ├── publisher/    发布器
│   ├── metrics/      指标导出
│   ├── server/       Unix Socket 管理接口
│   ├── archive/      帧归档（本地 JPEG）
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
