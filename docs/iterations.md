# 迭代记录

## Phase 1 — MVP

**完成**：2026-04-14

**目标**：单路 RTSP 软解 → TensorRT 推理 → Kafka 发布，跑通全链路。

**核心组件**：
- `FFmpegDecoder`：libavcodec 软解，`sws_scale` 转 BGR cv::Mat
- `FrameBuffer`：`boost::lockfree::spsc_queue<32>` 无锁单生产者单消费者队列
- `TRTBackend`：读取 `.engine`，CPU 预处理（resize + normalize），cudaMemcpy H2D，`executeV2`
- `KafkaPublisher`：librdkafka 异步生产，JSON 序列化，独立 publish 线程

**关键决策**：
- 推理输入走 CPU 预处理再 H2D，简单但存在两次内存拷贝
- FrameBuffer 容量 32 帧，超出则丢弃并计数

---

## Phase 2 — StreamPool + BatchScheduler

**完成**：2026-04-14

**目标**：多路并发，引入批处理调度。

**新增**：
- `StreamPool`：管理 N 路 `FFmpegDecoder` 实例，支持运行时增删流（`addStream` / `removeStream`）
- `BatchScheduler`：轮询所有 FrameBuffer，按 `max_batch=16` 或 `max_wait=10ms` 触发下发
- `InferWorker`：接收 Batch，串行调用 `IInferBackend::infer()` + `IYOLODecoder::decode()`

**关键决策**：
- BatchScheduler 以 500µs 轮询间隔忙等，避免条件变量延迟
- 每个 ModelConfig 对应独立的 InferWorker + BatchScheduler，多模型可并行

---

## Phase 3 — 多模型 YOLO 版本支持

**完成**：2026-04-14

**目标**：同时支持 YOLOv5 / v8 / v11 / v26，统一后处理接口。

**新增**：
- `IYOLODecoder` 抽象接口 + `DecoderFactory`
- `YOLOv5Decoder`：anchor-based，解析 `[bs, anchors, 5+nc]` 输出
- `YOLOv8Decoder`：anchor-free，解析 `[bs, 4+nc, 8400]` 输出（transpose + NMS）
- `YOLO11Decoder`：格式与 v8 相同，继承 `YOLOv8Decoder`
- `YOLO26Decoder`：格式待定，暂返回空结果并打印 WARN

**关键决策**：
- 后处理与推理后端解耦，Backend 只负责填充 `float[]`，Decoder 负责语义解析
- NMS 实现在 `IYOLODecoder` 基类，子类复用

---

## Phase 4 — NVDEC 硬件解码 + CUDA 预处理

**完成**：2026-04-14

**目标**：消除两次大 H2D memcpy，降低 CPU 占用，目标减少 ~8ms 延迟。

**改造路径**：
```
之前: FFmpeg 软解 → cv::Mat(CPU) → CPU resize+normalize → cudaMemcpy H2D → TRT
之后: FFmpeg NVDEC → NV12 GPU buf → fused CUDA kernel(resize+YUV→RGB+normalize) → TRT
```

**关键实现**：
- `GpuBuffer`：持有 `void* y_data / uv_data`（设备指针）+ `shared_ptr<void> frame_ref` 保持 AVFrame 引用计数
- `FFmpegDecoder`：检测 `h264_cuvid` / `hevc_cuvid` decoder；`av_frame_ref` 延长 NV12 surface 生命周期，`shared_ptr` 析构时释放
- `CudaPreprocess.cu`：16×16 线程块，每线程处理一个输出像素，双线性插值 + BT.601 YCbCr→RGB + normalize
- `TRTBackend`：新增 `infer_stream_`（cudaStream），`inferGPU()` 调用 kernel 后 `cudaStreamSynchronize` 再 `executeV2`
- `BatchScheduler`：按 `model_id` 过滤流（`StreamPool::getStreamModelId()`），同一批次强制同质（全 GPU 或全 CPU）

**踩坑**：
- NVDEC 帧指针是 FFmpeg buffer pool 管理的，`av_frame_unref` 后立即失效。必须 `av_frame_ref` 持引用，在 Batch 消费完毕后才能释放
- `executeV2` 走默认流，与 `infer_stream_` 异步，必须先 `cudaStreamSynchronize` 确保 kernel 写完再让 TRT 读输入 buffer

**配置**：`streams[].use_hwdec: true` 启用 NVDEC，`false` 退回软解（Ascend 机器默认 false）

---

## Phase 5 — Ascend 310P 后端

**完成**：2026-04-14

**目标**：适配华为 Ascend 310P NPU。

**核心约束**：310P 不支持动态 shape，需预编译多档 `.om`。

**实现**：
- `AscendBackend`：`loadModel()` 读取 `om_paths` map，按 batch=1/4/8/16 加载
- `selectModel(bs)`：选最接近且 ≥ bs 的档位，不足时 pad 输入
- `convert_ascend.sh`：调用 `atc` 工具批量生成 4 个档位的 `.om`
- `docker-compose.ascend.yml`：挂载 `/dev/davinci*` 设备

**状态**：框架完成，待真机联调。

---

## Phase 6 — 生产化（Prometheus + HTTP 管理）

**完成**：2026-04-14

**目标**：可观测性 + 热更新 + 优雅关闭。

**新增**：
- `Metrics`：自研轻量 Prometheus 文本格式导出器（无外部依赖）
  - `LabeledHistogram`（延迟）/ `LabeledCounter`（帧数、Kafka 消息数）
  - 直方图桶上界：1/2/5/10/20/50/100/200/500/1000 ms
- `ManagementServer`：cpp-httplib（header-only，FetchContent 引入）
  - `GET /healthz` — liveness probe
  - `GET /metrics` — Prometheus 格式
  - `GET/POST/DELETE /streams` — 运行时增删摄像头流
- Grafana 自动 provisioning：挂载 `config/grafana/provisioning/` 目录

**埋点位置**：
- `InferWorker`：`infer_latency_ms`（单批推理）、`e2e_latency_ms`（端到端）、`infer_batches_total`
- `FFmpegDecoder`：`frames_decoded_total`、`frames_dropped_total`
- `KafkaPublisher`：`kafka_published_total`、`kafka_dropped_total`

**优雅关闭顺序**：`ManagementServer::stop()` → Scheduler/Worker stop → `StreamPool::stopAll()` → `KafkaPublisher::flush()`

---

## Phase 7a — 性能优化（借鉴 Triton）

**完成**：2026-04-15

**目标**：消除推理热路径上的内存分配开销，支持多实例并行，预处理与推理流水线重叠。

**新增**：
- `GpuBufferPool`：预分配 `buffer_pool_size` 个 GPU buffer slot，`acquire()` / `release()` 无锁 CAS，替代每次推理的 `cudaMalloc` / `cudaFree`
- `InferWorkerGroup`：同一模型启动 N 个 InferWorker 实例（类 Triton `instance_group`），`enqueue()` 原子 round-robin 分发，配置项 `instance_count` / `device_ids`
- 异步双缓冲：`TRTBackend` 拆分为 `preprocess_stream_` + `infer_stream_`，用 `cudaEvent` 同步，预处理与上一批推理重叠；TRT 8.5+ 走 `enqueueV3`，旧版退回 `executeV2`

**配置项**：`buffer_pool_size`（默认 4）、`instance_count`（默认 1）、`preferred_batch_sizes`（Triton 风格凑批）、`max_queue_delay_us`（替代硬编码 10ms）

---

## Phase 7b — 功能扩展（借鉴 Triton + DeepStream）

**完成**：2026-04-15

**目标**：运行时热管理模型；主检测模型触发二级分类，属性注入原始检测结果后统一发 Kafka。

**新增**：
- `ModelManager`：模型生命周期状态机（`UNLOADED → LOADING → READY → DRAINING`），支持 `load` / `unload` / `swap`（先起新再停旧，重叠 < 30ms）
- `ManagementServer` 新增 `/models` REST 端点（GET / POST / DELETE / PUT / stats）
- 级联管道（DeepStream Primary/Secondary GIE 风格）：
  - `CascadeRouter`：按 `trigger_classes` 过滤检测框，GPU/CPU 裁剪 ROI → 二级 mini-batch
  - `ClassifierDecoder`：实现 `IYOLODecoder`，argmax 输出 top-1 Detection
  - `AttributePublisher`：二级模型专用 `IPublisher`，结果写入 `ResultMerger`
  - `ResultMerger`：等待当帧所有二级结果到齐（或超时），合并 `Detection.attributes` 后发布 Kafka

**关键决策**：
- `registerPrimary()` 在所有 `enqueue()` 之前调用，避免二级结果先到时找不到 pending entry
- 先统计全部 cascade 配置的 total_crops 再注册，防止多 cascade 时 key 碰撞
- `stopPipeline_unlocked` 持有 `CascadeOwnership`，确保 `ResultMerger` 在二级 worker 停止后再析构

---

## Phase 8 — 可选目标追踪（ByteTrack + DeepSORT 占位）

**完成**：2026-04-15

**目标**：在不破坏现有 Kafka 消费的前提下，为每路流提供可选目标追踪能力。

**新增**：
- `streams[].tracker` 配置项：`none` / `bytetrack` / `deepsort`
- `TrackerManager`：按 `stream_id` 维护追踪器状态，避免多 worker 并发下轨迹串线
- `ByteTrackTracker`：基于 IoU 的两阶段匹配（高分框 + 低分框）输出 `track_id`
- `InferWorker`：在发布前接入可选追踪逻辑，cascade 路径共享同一结果
- `KafkaPublisher`：按需输出 `detections[].track_id`（可选字段）

**状态**：
- `bytetrack`：可用
- `deepsort`：配置可识别，运行时提示未实现并跳过追踪（需后续接 ReID）

---

## Phase 9 — 帧归档并行化（Local + MinIO）

**完成**：2026-04-15

**目标**：检测发布不被文件 I/O 或云上传阻塞；单事件中携带可追溯的帧位置信息。

**新增**：
- `frame_archive` 配置块：本地目录、采样间隔、JPEG 质量、队列容量、MinIO 连接参数
- `FrameArchiver`：异步队列写盘，按需上传 MinIO（后台线程）
- `InferResult` 新字段：`frame_local_path` / `frame_url` / `frame_upload_state`
- `KafkaPublisher`：单事件输出 frame 元数据，不等待上传结果

**关键决策**：
- 主推理线程只做归档任务入队，不做写盘与上传，保持检测路径低延迟
- 单事件发布默认最终一致：MinIO 开启时 `frame_upload_state=pending`，失败仅记指标
- MinIO key 按 `stream_id/timestamp_frameSeq.jpg` 幂等命名，便于重放与追溯

---

## Phase 10 — 稳定性与可观测性

**完成**：2026-04-16

**目标**：让下游区分三种沉默（引擎挂 / 摄像头断 / 无目标）；消除重连风暴；补全流级可观测性。

**新增**：
- `StreamHealthRegistry`：全局单例状态机，`CONNECTING → STREAMING → RECONNECTING → DEGRADED → STOPPED`，`FFmpegDecoder` 在关键点调用
- `HeartbeatPublisher`：独立后台线程，每 5s 向 `inference-heartbeat` topic 发心跳，含 `stream_state` / `consecutive_failures` / `frames_since_last_hb`；引擎级心跳（`stream_id=null`）作为进程存活信号
- `Metrics` 新增流级 gauge：`stream_state` / `stream_reconnect_count` / `stream_consecutive_failures` / `stream_last_frame_age_seconds`（在心跳循环中刷新）
- `ManagementServer` 新增 `GET /health`（readiness probe，200/207/503）和 `GET /streams/{id}/health`
- 指数退避重连：`delay = min(base × 2^failures, max_reconnect_delay_ms)` + ±10% jitter

**关键决策**：
- `HeartbeatPublisher` 使用独立 Kafka producer，不复用 `KafkaPublisher`（语义分离，独立 topic）
- Metrics 刷新放在心跳循环，不在解码热路径上操作锁
- `/healthz` 保持无条件 200（K8s liveness），`/health` 作为 readiness probe

---

## Phase 11 — CPU / MPS 推理后端（ONNX Runtime）

**完成**：2026-04-16

**目标**：无 GPU 的开发机和 CI 环境可跑完整链路，验证 decoder / pipeline 不再依赖硬件。

**新增**：
- `OnnxBackend`：实现 `IInferBackend`，CPU EP 跨平台可用，MPS EP 走 CoreML（macOS Apple Silicon，`BUILD_ONNX_BACKEND_COREML=ON`，未编译时自动降级 CPU）
- `DeviceType::MPS`、`ModelConfig.onnx_path`
- `CMakeLists.txt`：`BUILD_ONNX_BACKEND` / `BUILD_ONNX_BACKEND_COREML` 选项；`find_package` 优先，找不到自动 FetchContent 下载 ORT v1.18.0
- `docker/Dockerfile.cpu` + `docker/docker-compose.cpu.yml`：无 CUDA 依赖的 CPU 推理栈

**关键决策**：
- MPS 不提供 Docker 镜像（macOS 容器无法访问 Metal GPU）
- 预处理与 `AscendBackend` 一致，现有 decoder 零改动

---

## Phase 12 — DAG Pipeline（sources / pipeline 模板 / tasks）

**完成**：2026-04-16

**目标**：将固定式 `streams -> model_id` 的链路升级为可编排 DAG pipeline，支持分支并行与汇合（例如 `decode -> {archive, infer} -> join`）。

**新增**：
- 新配置格式：`sources`（输入源）+ `pipelines`（仅 nodes/edges 的图模板）+ `tasks`（`source_id` + `pipeline_id`，运行实例）
- 运行时：`TaskManager` + `GraphExecutor` + `EdgeQueue`（每条边独立背压策略；每个 task 一份 executor）
- Stage：`source.rtsp` / `infer.engine` / `archive.raw` / `track.bytetrack` / `join.byFrameId` / `sink.kafka`（其余为占位 passthrough）
- 管理接口：`GET /tasks`、`POST /tasks/{id}/start|stop`

**关键决策**：
- `infer.engine` 对 ONNX Runtime 默认按 **batch=1** 运行（避免 ORT 输入 shape 自省不稳定；且多数 YOLO ONNX 为静态 batch=1）
- 模型加载采用 **lazy-load**（首次推理时加载），避免启动阶段的后端初始化不确定性

---

## Phase 13 — ByteTrack V2（stream 级参数化）

**完成**：2026-04-16

**目标**：将简化版 IoU 贪心追踪升级为更稳定的 ByteTrack V2 风格实现，并支持按 `streams[]` 单独配置追踪阈值与生命周期参数。

**新增**：
- `ByteTrackConfig`：新增 stream 级追踪参数（阈值、确认门槛、丢失清理）
- 配置接入：YAML 与 `POST /streams` 均支持 `tracker_params.bytetrack.*`
- 追踪内核：两阶段关联 + 线性分配 + `Tracked/Lost/Removed` 状态机
- 运行时更新：`TrackerManager` 检测 stream 参数变化后重建 tracker

**关键决策**：
- `min_hits_to_confirm` 生效后仅确认轨迹输出 `track_id`（默认 `2`）
- 统一使用 `validateByteTrackConfig()` 做配置校验，保持 YAML/REST 行为一致
- 修复速度预测更新振荡问题，提升多帧稳定性

**测试与验证**：
- 配置测试覆盖默认值、显式值、非法参数
- 追踪测试覆盖确认门槛、全局匹配与配置热更新
- `ctest --output-on-failure -R "(config|tracker)"`、`scripts/validate-repo.sh` 均通过

---

## Phase 14 — 断流产品化（终态 + Control Topic）

**完成**：2026-04-16

**目标**：上游流中断时不再"无限重连且语义模糊"，而是做到可终止、可观测、下游可明确区分。

**新增**：
- `StreamState` 新增终态 `FAILED`
- `sources[].degraded_threshold`：连续失败达到阈值进入 `DEGRADED`
- `sources[].max_reconnect_attempts`：连续失败达到阈值进入 `FAILED`，并停止重连（需手动/API 介入恢复）
- `KafkaConfig.control_topic`（默认 `inference-control`）：发送控制事件
  - `stream_dropped` / `stream_recovered` / `stream_failed_terminal`

**关键决策**：
- 达到 `max_reconnect_attempts` 后进入终态并停止重连，避免长期无效重试与噪音
- 控制事件独立 topic，不污染主推理结果流

---

## Phase 15 — sink.ffplay 本地实时预览

**完成**：2026-04-17

**目标**：开发调试阶段可在本机直接看到带检测框的实时视频。

**新增**：
- `SinkFfplayStage`：`popen` 启动 ffplay 子进程，以 pipe 写入原始 BGR24 帧；ffplay 退出后自动重连
- `DetectionOverlay`：`mapDetectionsFromModelToFrame`（模型坐标系 → 帧像素坐标）+ `overlay::drawDetections`（绘制 bbox / label / score）
- `DrawAndStreamStage`：叠图后送 ffplay，配置项 `draw_conf_thresh` / `line_thickness`
- DAG 新增 stage 类型：`sink.ffplay`

**关键决策**：
- ffplay 以 `-f rawvideo -pixel_format bgr24` 接收裸流，避免软件编码开销
- `mapDetectionsFromModelToFrame` 提取为独立工具函数，`InferWorker` 与 `InferEngineStage` 共享同一实现

---

## Phase 16 — InferEngineStage 路径补全 + 延迟指标闭环

**完成**：2026-04-20

**目标**：Phase 12 引入 DAG 后，`InferEngineStage` 的坐标映射与指标覆盖未与 `InferWorker` 对齐，本次补全并修复指标正确性问题。

**新增**：
- `infer_queue_latency_ms` histogram：帧从采集到进入推理批次的等待时间，主路径与 fallback（bs=1）均覆盖
- `e2e_latency_ms`：从 `capture_mono_ns` 到推理+解码完成的端到端延迟，写入 `InferResult` 并上报 Metrics
- `mapDetectionsFromModelToFrame`：补回 `InferEngineStage` 两条路径（合并冲突时意外删除）
- `recordInferBatchSize`：补回 `InferWorker`（误删后 `InferWorkerGroup` / cascade 路径 batch size 指标静默丢失）

**关键决策**：
- `batch_start_mono_ns` 提到循环外，消除每帧重复计算
- 所有 uint64_t 延迟减法前加 `>=` 判断：跨 NUMA 节点时 `capture_mono_ns` 可能略大于参考时间戳，无符号下溢会产生 ~1.8×10¹³ ms 的异常值

---

## Hotfix — TRTBackend tensor name discovery & inference safety

**完成**：2026-04-21

**目标**：修复 `feat: enhance TRTBackend to support dynamic input shapes and tensor name caching` 引入的若干回归问题。

**修复内容**：
- **CRITICAL**：恢复 `inferGPU()` 和 `infer()` CPU 路径中无条件调用 `setInputShape`。原条件守卫 `if (input_shape_dynamic_)` 会导致静态引擎在 `bs < max_batch_size_` 时读取未初始化 GPU 内存；对静态引擎而言 `setInputShape` 是 no-op，无条件调用是安全的。
- **HIGH**：`input_tensor_name_` 不再硬编码为 `"images"`，改为在 `loadModel()` 遍历 IO tensor 时自动发现，与 output 名称的发现逻辑对齐。非标准输入名的引擎不再静默失败。
- **HIGH**：移除 `enqueueV3` 分支中无用的 `bindings[2]` 数组及 `(void)bindings` 抑制，`enqueueV3` 使用 `setTensorAddress` 而非 bindings 数组。
- **MEDIUM**：`getTensorShape` 返回 `nbDims <= 0` 时抛出明确错误，而非静默跳过动态维度检测。
- **MEDIUM**：CPU 路径 `executeV2` 前补充 `setTensorAddress`，与 GPU 路径对齐，为 TRT 10 弃用 bindings 数组 API 做准备。
- 提取 `hasDynamicDims()` 至 `include/infer/TrtUtils.h`（无 TRT 依赖），新增 `tests/test_trt_utils.cpp` 8 个无 GPU 单元测试。

---

## Phase 17 — 帧采样模式重构（TimeBased / FrameCount）

**完成**：2026-04-22

**目标**：修复旧采样逻辑在非 25fps 源上的帧率偏差，引入基于 PTS 的时间戳采样模式，满足跨帧率源统一输出帧率的需求。

**旧问题**：
1. `sample_interval = 25 / sample_fps` 固定假设源为 25fps，30fps 源请求 5fps 实际输出 6fps
2. interval 在 `openStream()` 之前计算，无法获知真实帧率

**新增**：
- `SamplingMode` 枚举（`FrameCount` | `TimeBased`），通过 `sampling_mode` 字段在 `StreamConfig` / `TaskConfig` 中配置
- **FrameCount**（默认）：`interval = round(actual_fps / sample_fps)`，在 `openStream()` 成功后从 `avg_frame_rate` 计算；每次重连重置 `frame_seq_` 保持 phase 对齐
- **TimeBased**（新模式）：`frame_pts_sec - last_emit_pts >= 1.0 / sample_fps`，使用 `frame->pts × av_q2d(time_base)`；PTS 无效时自动退回 FrameCount
- `computeSampleInterval()` 提取为独立自由函数，无 FFmpeg 依赖，可单元测试
- `SamplingParams::shouldEmit()` 封装每次重连状态，`SourceRtspStage` / `SourceFileStage` / `StageFactory` / `TaskManager` 全链路传递 `sampling_mode`
- `avdevice_register_all()` 移至 `openStream()` 内调用，支持 `lavfi:` 虚拟源 URL（测试用）

**测试**：`test_frame_sampling.cpp` 16 个测试（12 单元 + 4 集成）全绿；`test_ffmpeg_file_decoder.cpp` 修复预存竞态条件（`waitForCompletion()` helper）。

**关键决策**：
- TimeBased 不依赖系统时钟，完全基于流内 PTS，视频文件循环回绕时不会产生采样跳变（`836f91c` 修复的 wall-clock 回归问题）

---

## Phase 18 — 多路发布扇出（gRPC + Redis Streams + MultiPublisher）

**完成**：2026-04-22

**目标**：支持推理结果同时下发到多个异构消费端（Kafka / gRPC / Redis），消费端按需订阅，不修改推理热路径。

**新增**：
- **`MultiPublisher`**：扇出至所有启用的 publisher，每个子 publisher 独立 try-catch 隔离，单路失败不影响其余路
- **`GrpcPublisher`**：实现 `InferenceService::Subscribe` server-streaming RPC，按 `stream_id` 过滤帧；慢订阅者超过 `kMaxPendingFrames=64` 时主动丢帧（防止单个慢客户端拖垮服务端）；`shutdown()` 等待 gRPC 客户端断连后返回（`4b4a701` 修复的阻塞关闭问题）
- **`RedisPublisher`**：`XADD` 到 Redis Streams，内部队列异步写（与 `KafkaPublisher` 相同模式），maxlen 可配；`pkg-config` fallback 兼容无 CMake 配置文件的旧版 gRPC（`56bf01c`）
- **`proto/inference.proto`**：定义 `InferenceService`（`Subscribe` RPC）+ `InferenceResult` / `BoundingBox` / `Detection` message
- **新配置块**：
  ```yaml
  publishers:
    kafka:    { enabled: true, ... }
    grpc:     { enabled: false, port: 50051 }
    redis:    { enabled: false, addr: "localhost:6379", stream_key: "infer:results" }
  ```
  旧 `kafka:` 根键自动 fallback 至 `publishers.kafka`，零迁移成本
- **`buildPublisher()`** 工厂：只有一路启用时直接返回单 publisher，多路时包装为 `MultiPublisher`
- **用户示例**：`examples/grpc_subscriber/`（Python + Go gRPC 订阅端示例，`6187f21`）

**测试**：`test_multi_publisher`（扇出、部分失败隔离、并发安全）、`test_grpc_publisher`（订阅/接收、stream-id 过滤、慢订阅者丢帧）、`test_redis_publisher`（stream key 格式、JSON payload、maxlen、队列丢帧、flush/drain）、`test_config`（新增 5 个 publishers 配置用例）全绿。

**构建选项**：`-DBUILD_GRPC_PUBLISHER=ON`（需 gRPC）、`-DBUILD_REDIS_PUBLISHER=ON`（需 hiredis）

---

## Phase 19 — GPU OOM & 引擎崩溃自愈

**完成**：2026-04-22

**目标**：不重启进程，在进程内完成 GPU 故障自愈；最坏情况下单 Worker 停止，其余流继续运行。

**新增**：
- **CRITICAL**: `include/infer/GpuFault.h` — `GpuFaultType` 分类枚举 + `GpuFaultException` / `GpuMemoryPressureException`；`classifyGpuError(int)` 将 CUDA 错误码映射到故障类型（无 CUDA header 依赖，可单元测试）
- **CRITICAL**: `include/pipeline/BatchPolicy.h` — 三级 batch 降级状态机（16→8→4→1），header-only 纯逻辑，`kRestoreThreshold=120` 次连续成功后升档
- **HIGH**: `GpuBufferPool` — 新增 `MemoryChecker` 可注入接口 + `isSafe()` 方法（默认 85% 水位检查）；`acquire()` 超水位时抛 `GpuMemoryPressureException` 实现预防性背压
- **HIGH**: `InferWorker` — 新增 `WorkerState` (RUNNING/RECOVERING/STOPPED)、`BatchPolicy`、`recoverFromFault()` 三路恢复：OOM→背压重入队、ENGINE_FAULT→engine reload、CONTEXT_LOST→cudaDeviceReset+reload；重试间隔 1s/2s/4s，3 次失败后标记 STOPPED
- **HIGH**: `InferWorkerGroup::enqueue()` — 跳过 RECOVERING/STOPPED worker，回退至 RECOVERING worker 而非直接丢弃
- **MEDIUM**: `TRTBackend` — `CUDA_CHECK` 宏升级为抛 `GpuFaultException`，新增 `checkInferTimeout()`（默认 5s）
- **MEDIUM**: 5 个新 Prometheus 指标：`gpu_oom_total`、`gpu_engine_fault_total`、`infer_worker_state`、`infer_batch_size_current`、`gpu_memory_usage_ratio`

**测试** (`tests/test_gpu_fault_recovery.cpp`，16 个测试全绿）：
- BatchPolicy 状态机（6 个）：onOOM 降级、onSuccess 升档、isExhausted 边界
- classifyGpuError 映射（3 个）：OOM / CONTEXT_LOST / ENGINE_FAULT
- 新 Metrics 断言（4 个）：Prometheus 序列化输出验证
- InferWorker 状态机集成（3 个）：OOM 保持 RUNNING + 指标 +1、ENGINE_FAULT 触发 reload 恢复 RUNNING、连续 reload 失败→STOPPED

**关键决策**：
- OOM 时 batch 放回队列头（`enqueueHead`）而非丢弃，减少漏检
- `BatchPolicy` 与 GPU 完全解耦（header-only 纯逻辑），无 CUDA 依赖可单元测试
- `MemoryChecker` 可注入，测试无需真实 GPU
- 仅 `CONTEXT_LOST` 才触发 `cudaDeviceReset`，普通崩溃只 reload engine

---

## Hotfix — InferEngineStage 背压死锁 + 低帧率积压

**完成**：2026-04-23

**目标**：修复 DAG 生产环境两个严重问题：低采样率（5fps）下 `queue_latency` 高达 700-900ms；下游 `EdgeQueue` 满时持锁推理导致整个 stage 冻结 1-2 秒。

**修复内容**：
- **CRITICAL**：新增 `flushLoop()` 后台线程，每 `max_queue_delay_us / 2` 唤醒一次，若 deadline 已过期且没有新帧到达则强制触发 batch flush。旧逻辑 deadline 仅在 `process()` 内检查，低帧率下两帧间隔远超 deadline 窗口。
- **CRITICAL**：两锁分离设计——`pending_mutex_` 只保护 `pending_events_` / `batch_deadline_` / `last_emit_`（持有时间 ≤ 微秒级，绝不跨 I/O 或 GPU 调用）；`infer_mutex_` 独立序列化 GPU 推理，防止 `process()` 与 `flushLoop()` 并发推理。旧设计在 `doFlush()` 全程持 `pending_mutex_`，下游满载时 `emit()` 阻塞会连带冻结入队路径。
- **HIGH**：`InferEngineStage` 补填 `Detection.class_name`（在 `InferWorker` 中存在但 DAG 路径遗漏，导致 Kafka 输出中 class_name 为空）。
- **MEDIUM**：`8c96349` 补充每帧入队延迟 debug 日志（`queue_latency_ms`），便于在生产环境定位积压。

**关键决策**：
- `process()` 和 `flushLoop()` 均调用 `extractBatch()`（加 `pending_mutex_`）后立即释放，再进入 `inferAndEmit()`（加 `infer_mutex_`），两阶段完全解耦
- `flushLoop()` 检查间隔取 `max_queue_delay_us / 2` 而非固定值，保证最坏延迟 ≤ 1.5× 配置值

---

## Phase 9 补丁 — GPU 帧归档 + FrameArchiver 多 Worker

**完成**：2026-04-24

**目标**：补齐 NVDEC 硬解路径的归档缺失，并提升高写盘压力下的归档吞吐量。

**修复与新增**：
- **`c9f88d2` — GPU 路径归档**：`InferWorker` 原有 `!batch.is_gpu` 守卫导致 GPU 批次静默跳过归档。新增 `nv12GpuToBgr()` helper（仅 `BUILD_TRT_BACKEND`），通过 `cudaMemcpy` 下载 NV12 Y/UV plane 后 `cv::cvtColor(COLOR_YUV2BGR_NV12)` 转换为 BGR `cv::Mat`，供 `FrameArchiver::submit` 使用
- **`18430cd` — ArchiveRawStage GPU 支持**：DAG 路径的 `ArchiveRawStage` 新增 `allow_gpu_frames` 配置（`frame_archive.allow_gpu_frames: true`），process 方法对 GPU 帧做相同 NV12→BGR 下载并提交，含 fallback 降级逻辑；补充完整 CPU/GPU 路径单元测试
- **`5594104` — 落盘计时日志**：worker loop 记录单帧写盘耗时（`write_ms`）与弹出后队列深度（`queue_depth_after_pop`），便于定位写盘瓶颈（日志示例：`FrameArchiver: wrote /data/... write_ms=9 queue_depth_after_pop=2224`）
- **`6535377` — 多 Worker 支持**：`FrameArchiveConfig` 新增 `worker_count`（默认 1，建议 ≤ CPU 核数），`FrameArchiver` 启动对应数量的后台写盘线程并行消费队列；配置项 `frame_archive.worker_count: 4`

**关键决策**：
- 多 worker 并发消费同一 `BlockingQueue`，每帧只被一个 worker 写盘，无需额外同步
- `allow_gpu_frames` 默认关闭，避免在无 CUDA 环境（CPU/Ascend 机器）编译报错

---

## 诊断增强 — EventEnvelope 上下文字段

**完成**：2026-04-24

**目标**：在不修改推理逻辑的前提下，为每帧事件附加队列上下文，便于在日志中快速定位背压位置。

**新增字段**（`include/pipeline/Event.h`）：
- `source_queue_size`（`SourceRtspStage` 出口填充）：源队列当前深度，反映采集端积压
- `ingress_edge` / `ingress_edge_queue_size`（`GraphExecutor` 填充）：事件经过的入边名称与该边队列深度
- `received_at_infer_ns`（`InferEngineStage` 入队时填充）：事件进入推理 stage 的单调时钟时间戳，与 `capture_mono_ns` 做差即得 transit time（采集→推理队列的链路延迟）

**日志输出**：`InferEngineStage` debug 日志增加 `transit_ms`（`received_at_infer_ns - captured_at_ns`）和 `pending_ms`（帧在推理队列中等待的时间），单独可见两段延迟分布。

---

## Phase 20 — DAG infer.engine 多实例/多卡 + 多 GPU NVDEC 负载均衡

**完成**：2026-04-25

**目标**：`pipelines` / `tasks` 中的 `infer.engine` 使用 `InferWorkerGroup`，使 `models[].instance_count` 与 `models[].device_ids` 在编排路径下生效；同时引入自动多 GPU NVDEC 负载均衡，100 路摄像头无需手动配置 device_id。

**新增**：
- `InferEngineWorkerStage`（`infer.engine`）：沿用 `InferEngineStage` 的 pending + deadline flush，将 `Batch` 交给 `InferWorkerGroup`；通过内部 `IPublisher` 桥接将 `InferResult` 还原为带 `infer_result` 的 `EventEnvelope` 并 `emit`（按 `stream_id` + `frame_meta.frame_seq` 关联，支持多 worker 乱序完成）
- `StageFactory`：`infer.engine` 创建该 stage（保留 `InferEngineStage` 源码供对照）
- `GpuDeviceAllocator`：线程安全最小负载 GPU 分配器，`acquire()` / `release()`，count 不允许为负，空列表时抛出；`StageFactory::Context` / `TaskManager` 从所有 model `device_ids` 取并集构建，传递给 source stages
- `GpuBuffer` / `Batch.cuda_device_id`：originating CUDA 设备号贯穿完整 pipeline
- `FFmpegDecoder`：将 device string 传给 `av_hwdevice_ctx_create`，每帧打 `cuda_device_id`；`SourceRtspStage` 在 `start()` 时 acquire 设备，`stop()` 时 release
- `InferWorkerGroup::enqueue(batch, restrict_cuda_device_id)`：GPU batch 路由到同设备 worker；无匹配时 fallback 至任意 RUNNING worker
- `BatchScheduler`：从第一个 GPU 帧继承 `batch.cuda_device_id`

**测试**：`test_infer_engine_worker_stage.cpp`（FakeBackend 校验 `device_ids`）；`test_stage_factory.cpp`（`BUILD_ONNX_BACKEND` 下 `dynamic_cast` 校验）；`GpuDeviceAllocator` 7 个单元测试（fairness / release / edge cases）全绿。

**状态**：✅ 完成

---

## Phase 21 — Ascend NPU 性能优化（BufferPool / 异步推理 / AIPP / DVPP）

**完成**：2026-04-26

**目标**：消除 Ascend 推理热路径上的逐次 `aclrtMalloc` / `aclrtFree` 开销；利用 AIPP 硬件预处理跳过 CPU normalize；引入 DVPP 硬件解码实现端到端零拷贝路径。

**新增**：
- **`AscendBufferPool`**（`include/infer/AscendBufferPool.h` + `src/infer/AscendBufferPool.cpp`）：预分配 `pool_size=4` 个 NPU HBM 槽位，lock-free CAS `acquire()` / `release()`（镜像 `GpuBufferPool`）；`MemoryChecker` 可注入，`initForTest()` 用 `operator new` 替代 `aclrtMalloc`，无需真实 NPU
- **`aclmdlExecuteAsync`**（`AscendBackend`）：将同步 `aclmdlExecute` 替换为异步 + `aclrtSynchronizeStream`，激活 `loadModel()` 中一直未使用的 `stream_`；RAII `SlotGuard` 确保异常路径也能释放 buffer slot；`AclExecuteAsyncFn` / `AclSyncStreamFn` 可注入 stub
- **AIPP 自动检测**：`loadModel()` 时调用 `aclmdlGetFirstAippInfo` 探测 `.om` 是否含 AIPP 算子；AIPP 路径跳过 `preprocessCPU`，改用 `packBgrUint8()` 打包 HWC BGR uint8，输入 buffer 缩至 `H×W×3` 字节（vs float 路径的 `×4`）；`AippDetectFn` 可注入 stub
- **`DVPPDecoder`**（`include/stream/DVPPDecoder.h` + `src/stream/DVPPDecoder.cpp`）：实现 `IStreamDecoder`，FFmpeg 仅做 demux，压缩包直送 DVPP 硬件解码；解码输出 YUV420SP 留在 NPU HBM（`Frame.ascend_buf`）；AIPP + DVPP 组合时 CPU 零拷贝：DVPP → AIPP in `.om` → 推理；`DvppApiStub` 可注入
- **新配置项（任务级）**：`TaskConfig.use_ascend_dvpp` / `TaskConfig.ascend_device_id`
- **新类型**：`Types.h` 新增 `AscendBuffer` 结构体 + `Frame.is_ascend` 标记位
- **`SourceRtspStage`**：检测 `use_ascend_dvpp` 后在 `start()` / `stop()` 中创建/释放 `DVPPDecoder`

**测试**（全部仅需 `BUILD_ASCEND_BACKEND`，无真机要求）：
- `test_ascend_buffer_pool.cpp`（9 个：pool 大小、acquire/release、OOM guard、并发安全）
- `test_ascend_async_infer.cpp`（4 个：调用顺序、异常安全）
- `test_ascend_aipp.cpp`（5 个：AIPP 路径 vs CPU 路径切换、输入 buffer 大小验证）
- `test_dvpp_decoder.cpp`（6 个：生命周期、`DvppApiStub` 注入）

**状态**：✅ 完成

---

## Phase 22 — DVPP → AIPP 零拷贝管道

**完成**：2026-04-27

**目标**：打通 DVPP 硬件解码到 AIPP 预处理的端到端零拷贝路径，彻底消除 CPU 预处理（`cv::resize`、`packBgrUint8`）和 H2D memcpy。

**背景**：Phase 21 已实现 `DVPPDecoder`（Frame.is_ascend=true），但 `Batch` 结构体没有 `ascend_frames` 字段，DVPP 输出的 HBM NV12 帧在 BatchScheduler 处被丢弃，AIPP 仍通过 `packBgrUint8`（CPU resize）喂数据。

**新增**：
- **`Batch.ascend_frames`**（`Types.h`）：`vector<AscendBuffer>` + `is_ascend` 标志，用于携带 HBM NV12 帧穿越 BatchScheduler 到 AscendBackend
- **`BatchScheduler`**：检测 `Frame.is_ascend`，将 `ascend_buf` 移入 `batch.ascend_frames`；扩展混合帧类型检测（CPU / GPU / Ascend 不可混批）；flush 后正确清零 ascend 字段
- **`InferEngineWorkerStage`**：图模式管道同步支持 Ascend 帧组批
- **`AscendBackend::infer()` 三路分支**：
  - **Path A（新）零拷贝**：`is_ascend && aipp_enabled_` → batch=1 时 DVPP device ptr 直传 ACL dataset（无任何拷贝）；batch>1 时 D2D memcpy 拼接到临时 HBM buffer（无 CPU 参与）
  - **Path B（原）CPU BGR**：`packBgrUint8` → H2D → AIPP
  - **Path C（原）CPU float**：`preprocessCPU` → H2D → NPU
- **异常安全**：`TmpDevGuard` 通过 `void**` 引用持有 `tmp_device`，保证 D2D concat 途中任意异常均能正确 `aclrtFree`
- **测试桩**：`setDevAllocFnForTest` / `setDevFreeFnForTest` 注入 device alloc/free，配合 `inferWithStubs()` 覆盖 ascend 路径

**激活方式**：
```yaml
tasks:
  - id: task_cam_001
    ...
    use_ascend_dvpp: true
    ascend_device_id: 0
    ...
```

---

## Phase 23 — DVPP CANN6 稳定性修复

**完成**：2026-04-27

**目标**：修复 DVPP 解码器在 CANN6 环境下的内存泄漏、发送失败后的资源回收缺失，以及 CANN6/CANN7 同步策略不一致问题。

**修复**：
- **回调死循环**（CRITICAL）：`start()` 将用户 callback 直接通过 `std::move` 传入 `decodeLoop`，不再创建中间 lambda；`decodeLoop` 在 `ctx_.cb` 中存储真实 callback，彻底消除 `onDecoded` → lambda → `ctx_.cb`（= lambda）的无限递归
- **`initChannel` 部分初始化资源泄漏**（HIGH）：`decodeLoop` 在 `initChannel()` 返回 false 后立即调用 `destroyChannel()`，释放已分配的 `channel_desc_` / `dvpp_stream_`（`destroyChannel` 对 null 指针安全）
- **CANN6 vs CANN7 同步**：CANN7 用 `aclrtSynchronizeStream(dvpp_stream_)`（精确到 vdec stream）；CANN6 `aclvdecSendFrame` 无 per-call stream，用 `aclrtSynchronizeDevice()`，在包被 unref 前保证 DMA 完成
- **发送失败回收**：`aclvdecSendFrame` / `acldvppVdecProcess` 失败时正确 free `yuv_buf` 并 destroy `pic_desc` / `stream_desc`，防止 DVPP 不持有描述符时内存泄漏

**状态**：✅ 完成

---

## 待办

- [x] Phase 23 DVPP CANN6 稳定性修复（内存释放 + 发送错误回收 + stream 同步）
- [ ] 任务级自动降级（安全实现待完善）：`use_ascend_dvpp=true` 且模型无 AIPP 时自动回退 CPU 解码
- [ ] Phase 4 真机 P99 延迟测试（目标 < 100ms @ 100路）
- [ ] Phase 5 Ascend 310P 真机联调
- [ ] CascadeRouter GPU 路径：从 secondary ModelConfig 读取实际 input_size（当前硬编码 112×112）
- [ ] Grafana 预置 Dashboard JSON（延迟热力图 + 丢帧率）
- [ ] 单元测试（Decoder NMS 逻辑、ClassifierDecoder argmax、ResultMerger 超时逻辑）
- [ ] YOLO26Decoder 格式确认并完成实现（当前返回空结果）
- [ ] deepsort 追踪器接入 ReID 模型
