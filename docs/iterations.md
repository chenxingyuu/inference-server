# 迭代记录

## Phase 1 — MVP

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

---

## Phase 7a — 性能优化（借鉴 Triton）

**目标**：消除推理热路径上的内存分配开销，支持多实例并行，预处理与推理流水线重叠。

**新增**：
- `GpuBufferPool`：预分配 `buffer_pool_size` 个 GPU buffer slot，`acquire()` / `release()` 无锁 CAS，替代每次推理的 `cudaMalloc` / `cudaFree`
- `InferWorkerGroup`：同一模型启动 N 个 InferWorker 实例（类 Triton `instance_group`），`enqueue()` 原子 round-robin 分发，配置项 `instance_count` / `device_ids`
- 异步双缓冲：`TRTBackend` 拆分为 `preprocess_stream_` + `infer_stream_`，用 `cudaEvent` 同步，预处理与上一批推理重叠；TRT 8.5+ 走 `enqueueV3`，旧版退回 `executeV2`

**配置项**：`buffer_pool_size`（默认 4）、`instance_count`（默认 1）、`preferred_batch_sizes`（Triton 风格凑批）、`max_queue_delay_us`（替代硬编码 10ms）

---

## Phase 7b — 功能扩展（借鉴 Triton + DeepStream）

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

## 待办

- [ ] Phase 4 真机 P99 延迟测试（目标 < 100ms @ 100路）
- [ ] Phase 5 Ascend 310P 真机联调
- [ ] CascadeRouter GPU 路径：从 secondary ModelConfig 读取实际 input_size（当前硬编码 112×112）
- [ ] Grafana 预置 Dashboard JSON（延迟热力图 + 丢帧率）
- [ ] 单元测试（Decoder NMS 逻辑、ClassifierDecoder argmax、ResultMerger 超时逻辑）

---

## Phase 8 — 可选目标追踪（ByteTrack + DeepSORT 占位）

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
