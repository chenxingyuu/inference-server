# 华为昇腾 (Ascend) 开发指南

本文档面向 C++ 推理服务开发者，梳理昇腾生态的核心概念、软件栈组成、版本体系以及实际开发要点。

---

## 目录

1. [硬件系列](#1-硬件系列)
2. [软件栈层次](#2-软件栈层次)
3. [Driver 驱动](#3-driver-驱动)
4. [Firmware 固件](#4-firmware-固件)
5. [CANN SDK](#5-cann-sdk)
6. [商用版与社区版](#6-商用版与社区版)
7. [版本配套关系](#7-版本配套关系)
8. [Docker 镜像体系](#8-docker-镜像体系)
9. [AscendCL 编程模型](#9-ascendcl-编程模型)
10. [内存模型](#10-内存模型)
11. [.om 离线模型](#11-om-离线模型)
12. [AIPP 预处理加速](#12-aipp-预处理加速)
13. [DVPP → AIPP 零拷贝管道](#13-dvpp--aipp-零拷贝管道)
14. [常见错误速查](#14-常见错误速查)
15. [日志与调试](#15-日志与调试)
16. [本项目适配要点](#16-本项目适配要点)

---

## 1. 硬件系列

### 芯片型号

| 系列 | 定位 | 典型型号 | 备注 |
|------|------|---------|------|
| 昇腾 310 | 推理 | 310, **310P**, **310P3** | 最常见推理卡 |
| 昇腾 910 | 训练 + 推理 | 910A, 910B, 910C | 大模型训练主力 |
| 昇腾 310B | 边缘推理 | 310B4 | 低功耗边缘场景 |

> **310P3** 是 310P 的工程量产版，推理性能 80 TOPS，常见于 Atlas 300I Pro 服务器。本项目目标硬件。

### 硬件形态

| 产品 | 规格 | 典型场景 |
|------|------|---------|
| Atlas 300I Pro | PCIe 推理卡，单卡 4×310P | 数据中心推理 |
| Atlas 300V Pro | 视频分析专用卡 | 视频结构化 |
| Atlas 500 | 边缘推理盒子 | 现场部署 |
| Atlas 800 | 推理服务器 | 大规模推理集群 |

---

## 2. 软件栈层次

```
┌─────────────────────────────────────────┐
│  业务应用 (inference-server)              │
├─────────────────────────────────────────┤
│  MindSpore / PyTorch (训练框架，可选)      │
│  AscendCL  (推理 C++ / Python API)       │
├─────────────────────────────────────────┤
│  CANN                                   │
│  ├── ATC   (模型编译器，ONNX → .om)       │
│  ├── GE    (图引擎，运行时优化)            │
│  ├── TBE   (算子库)                      │
│  └── DVPP  (视频/图像硬件加速)            │
├─────────────────────────────────────────┤
│  NPU Firmware (固件，烧录到芯片)          │
├─────────────────────────────────────────┤
│  NPU Driver  (内核模块)                  │
├─────────────────────────────────────────┤
│  Linux 内核 / 物理硬件                   │
└─────────────────────────────────────────┘
```

---

## 3. Driver 驱动

### 作用
内核模块，使操作系统识别 NPU 设备，安装后出现 `/dev/davinci*` 设备节点。

### 安装包命名
```
Ascend-hdk-310p-npu-driver_24.1.RC3_linux-aarch64.run
```

### 关键路径
```
/usr/local/Ascend/driver/         ← 驱动安装目录
/dev/davinci0 ~ /dev/davinci3     ← 单张 300I Pro 卡的 4 个设备
/dev/davinci_manager              ← 管理设备（必须挂载进容器）
/dev/devmm_svm                    ← 内存管理设备
```

### 注意事项
- 驱动版本必须与 CANN 版本严格配套，**不匹配是最常见的环境问题**
- 升级 Linux 内核后必须重新编译/安装驱动
- 容器内**不需要**安装驱动，但必须挂载宿主机设备节点

---

## 4. Firmware 固件

### 作用
烧录到 NPU 芯片内部的微代码，控制芯片底层行为。

### 安装包命名
```
Ascend-hdk-310p-npu-firmware_7.5.0.1.129.run
```

### 注意事项
- 固件与驱动通常**打包一起发布**，版本号绑定
- 升级固件后需要**重启服务器**才能生效
- 可通过 `npu-smi info` 查看当前固件版本

---

## 5. CANN SDK

CANN（Compute Architecture for Neural Networks）是昇腾最核心的开发套件。

### 主要组件

| 组件 | 说明 | 对应 CUDA 概念 |
|------|------|--------------|
| **AscendCL** | C/C++/Python 推理 API | cuDNN + CUDA Runtime |
| **ATC** | AI 编译器，将 ONNX/Caffe 转换为 .om | `trtexec` |
| **GE** | 图引擎，运行时图优化与调度 | TensorRT Engine |
| **TBE** | 算子库（Tensor Boost Engine）| cuBLAS/cuDNN 算子 |
| **DVPP** | 视频/图像硬件解码加速 | NVDEC/NVJPEG |
| **AIPP** | 模型内嵌预处理加速 | 无直接对应 |

### 安装包命名
```
Ascend-cann-toolkit_8.0.RC3_linux-x86_64.run    ← 开发工具包（含 ATC）
Ascend-cann-kernels-310p_8.0.RC3_linux.run       ← 算子包（运行时必需）
```

### 关键路径（容器/宿主机）
```
/usr/local/Ascend/ascend-toolkit/latest/
├── bin/atc                      ← 模型转换工具
├── include/acl/                 ← AscendCL 头文件
├── lib64/libascendcl.so         ← 运行时库
└── tools/profiler/              ← 性能分析工具
```

---

## 6. 商用版与社区版

| 维度 | 商用版 | 社区版 (Community) |
|------|--------|-------------------|
| 获取方式 | 华为官方渠道采购 | [昇腾社区](https://www.hiascend.com) 免费下载 |
| 版本号格式 | `8.0.0`, `8.1.0` | `8.0.RC1`, `8.0.RC3` |
| 稳定性 | 完整 QA 认证，生产就绪 | RC = Release Candidate，功能较新，可能有 bug |
| 技术支持 | 商业 SLA | 社区论坛 + issue |
| 功能差异 | 完整功能集 | 与商用基本一致，部分高级功能可能延后 |
| 适用场景 | 生产环境 | 开发测试、学习、POC |

> 本项目 Docker 镜像 `ascendai/cann:8.5.1-310p` 使用的是**社区版**。

---

## 7. 版本配套关系

驱动、固件、CANN **三者必须严格配套**，这是昇腾环境最容易出问题的地方。

### 版本号规律

```
社区版:  CANN 8.0.RC3  →  驱动 24.1.RC3
商用版:  CANN 8.0.0    →  驱动 24.1.0
```

### 典型配套表（310P / Atlas 300I Pro）

| CANN 版本 | 驱动版本 | 固件版本 | Docker 基础镜像 | 备注 |
|----------|---------|---------|----------------|------|
| 6.0.1 | 22.0.x | 6.x | `ascendai/cann:6.0.1-310p-ubuntu20.04-py3.9` | CANN 6，Ubuntu 20.04 |
| 8.0.RC3 | 24.1.RC3 | 7.3.0.1.231 | `ascendai/cann:8.0.RC3-310p-ubuntu22.04-py3.11` | 社区版 |
| 8.5.1 | 25.0.x | 7.5.x | `ascendai/cann:8.5.1-310p-ubuntu22.04-py3.11` | 社区版，默认 |

> 最新配套关系以[昇腾官方文档](https://www.hiascend.com/document)为准。

### 验证方法

```bash
# 查看驱动版本
cat /usr/local/Ascend/driver/version.info

# 查看 CANN 版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg

# 查看固件版本
npu-smi info | grep Firmware
```

---

## 8. Docker 镜像体系

### 官方镜像命名规则

```
ascendai/cann:8.5.1-310p-ubuntu22.04-py3.11
           │    │     │      │          │
           │    │     │      └── Python 版本
           │    │     └──────── 操作系统
           │    └────────────── 芯片型号
           └─────────────────── CANN 版本
```

### 镜像类型

| 镜像 | 包含内容 | 适用场景 |
|------|---------|---------|
| `ascendai/cann` | CANN 运行时 + AscendCL | 推理部署（本项目） |
| `ascendai/mindie` | CANN + MindIE 大模型推理框架 | LLM 推理 |
| `ascendai/mindspore` | CANN + MindSpore | 模型训练 |

### Dockerfile 与 docker-compose 选择

| 场景 | Dockerfile | compose |
|------|-----------|---------|
| CANN 8（默认） | `docker/Dockerfile.ascend` | `docker/docker-compose.ascend.yml` |
| CANN 6 / Atlas 300I Pro | `docker/Dockerfile.ascend.cann6` | `docker/docker-compose.ascend.300ipro.yml` |

构建 CANN 6 镜像：

```bash
docker build \
  -f docker/Dockerfile.ascend.cann6 \
  -t inference-server:ascend-cann6 \
  .
```

### docker-compose 设备挂载

Atlas 300I Pro 单卡包含 **4 个独立 NPU 设备**，compose 文件需挂载 davinci0~3：

```yaml
services:
  infer-ascend:
    devices:
      - /dev/davinci0           # NPU 计算设备
      - /dev/davinci1
      - /dev/davinci2
      - /dev/davinci3
      - /dev/davinci_manager    # 管理接口（必须）
      - /dev/devmm_svm          # 内存管理（必须）
      - /dev/hisi_hdc           # 调试接口
    volumes:
      - /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro  # 驱动库只读挂载
```

---

## 9. AscendCL 编程模型

### 对象层次

```
Device  (物理 NPU 卡，/dev/davinci0)
  └── Context  (执行上下文，线程独占)
        └── Stream  (命令队列，异步执行)
              └── Task  (推理 / 内存拷贝 / 算子)
```

### 与 CUDA 对比

| CUDA | AscendCL |
|------|----------|
| `cuInit()` | `aclInit(nullptr)` |
| `cudaSetDevice()` | `aclrtSetDevice(device_id)` |
| `cuCtxCreate()` | `aclrtCreateContext(&ctx, device_id)` |
| `cudaStream_t` | `aclrtStream` |
| `cudaStreamCreate()` | `aclrtCreateStream(&stream)` |
| `cudaMemcpy()` | `aclrtMemcpy()` |
| `cudaMalloc()` | `aclrtMalloc()` |
| `cudaMallocHost()` | `aclrtMallocHost()` |
| `.engine` | `.om` |

### 典型初始化流程

```cpp
// 1. 全局初始化（进程级，调用一次）
aclInit(nullptr);

// 2. 选择设备（线程级）
aclrtSetDevice(device_id);

// 3. 创建上下文（线程级，不可跨线程共享）
aclrtContext ctx;
aclrtCreateContext(&ctx, device_id);

// 4. 创建命令流
aclrtStream stream;
aclrtCreateStream(&stream);

// --- 推理循环 ---

// 5. 销毁（逆序）
aclrtDestroyStream(stream);
aclrtDestroyContext(ctx);
aclrtResetDevice(device_id);
aclFinalize();
```

### Context 创建规则（CANN 6 必须显式创建）

CANN 8 在 `aclrtSetDevice()` 时会自动创建一个默认 Context，CANN 6 **不保证**这一行为。
本项目在 `AscendBackend::loadModel()` 中始终显式调用 `aclrtCreateContext()`，兼容两个版本。

```cpp
aclrtSetDevice(device_id_);
aclrtCreateContext(&ctx_, device_id_);  // 必须在 CANN 6 下显式创建
aclrtCreateStream(&stream_);
```

### Context 线程绑定规则（重要）

Context 与创建它的线程**强绑定**，不能跨线程传递或使用。

```cpp
// 正确：每个 InferWorker 线程自己创建 Context
void InferWorker::threadFunc() {
    aclrtSetDevice(device_id_);
    aclrtCreateContext(&ctx_, device_id_);
    // 本线程内安全使用 ctx_
}

// 错误：在主线程创建 Context，传给子线程使用
// → 会触发 ACL_ERROR_RT_CONTEXT_NULL
```

---

## 10. 内存模型

### 内存类型

```
Host 侧 (CPU 内存)              Device 侧 (NPU 内存)
┌──────────────────┐           ┌──────────────────┐
│ aclrtMallocHost  │           │ aclrtMalloc      │
│ (页锁定，DMA 友好)│ ──H2D──▶ │ (NPU 显存)       │
│                  │ ◀──D2H── │                  │
└──────────────────┘           └──────────────────┘
         ▲
         │  普通 malloc 也可用，但性能差
```

### 内存拷贝方向枚举

```cpp
ACL_MEMCPY_HOST_TO_HOST    // CPU → CPU
ACL_MEMCPY_HOST_TO_DEVICE  // CPU → NPU
ACL_MEMCPY_DEVICE_TO_HOST  // NPU → CPU
ACL_MEMCPY_DEVICE_TO_DEVICE // NPU → NPU
```

### 性能建议

- 输入数据用 `aclrtMallocHost` 分配，避免额外拷贝
- 批量小数据优先用同步拷贝；大数据用异步拷贝 + stream 同步
- NPU 内部中间 buffer 用 `aclrtMalloc`，推理结束后释放

---

## 11. .om 离线模型

### 什么是 .om

`.om`（Offline Model）是 ATC 编译器针对特定昇腾芯片**预编译**的推理模型，类似 TensorRT 的 `.engine`。一旦编译，运行时无需再做图优化，直接执行。

### 转换流程

```
ONNX / Caffe / MindSpore 模型
            │
            ▼
        ATC 编译器
            │
            ▼
        .om 文件（绑定芯片型号 + 固定 batch size）
```

### YOLO 导出 ONNX（Ultralytics）

ATC 的输入是 ONNX。用 **Ultralytics** 从 `.pt` 导出时，需同时满足 **动态 batch**（否则按多 batch 编译时 DFL/Reshape 易形状不一致）和 **较低 ONNX opset**（新版默认 opset 18+ 甚至 20，老版本 ATC 会报 `No parser is registered for Op [... ai.onnx::20::Conv]`）。

在已安装 `ultralytics` 的环境执行（`batch` 取你计划编译的最大 batch，与 `scripts/convert_ascend.sh` 中 1/4/8/16 一致即可）：

```bash
yolo export \
  model=yolov8n.pt \
  format=onnx \
  dynamic=True \
  batch=16 \
  imgsz=640 \
  opset=12
```

| 参数 | 说明 |
|------|------|
| `dynamic=True` | 输入 batch 维动态，配合多个 `--input_shape` 分别 ATC |
| `batch=16` | 导出时图展开的上限，与最大 `.om` batch 对齐 |
| `opset=12` | 与常见 CANN 6 / 旧 ATC 兼容；若仍报无 parser，可试 `opset=11`；较新 CANN 可对照官方文档尝试更高 opset |
| `imgsz=640` | 与 `convert_ascend.sh` 里 `640×640` 一致 |

导出后默认输入名一般为 **`images`**（与仓库脚本、配置一致）。若自定义过导出，用 Netron 打开 ONNX 核对输入名再改 `convert_ascend.sh` 的 `--input_shape` 前缀。

接着在本机或已 `source .../set_env.sh` 的转换机上执行 `scripts/convert_ascend.sh <onnx> <name>` 生成各 batch 的 `.om`。

### ATC 转换命令

```bash
atc \
  --model=yolo11n.onnx \
  --framework=5 \                        # 5=ONNX, 0=Caffe, 1=MindSpore
  --output=yolo11n_bs8 \                 # 输出文件名（不含 .om 后缀）
  --input_shape="images:8,3,640,640" \   # 固定 batch=8
  --input_format=NCHW \
  --soc_version=Ascend310P3 \
  --precision_mode=allow_fp32_to_fp16    # 允许 FP32 算子转 FP16
```

### 常用 ATC 参数

| 参数 | 说明 |
|------|------|
| `--framework` | 源模型框架：5=ONNX, 0=Caffe, 1=MindSpore |
| `--soc_version` | 目标芯片：`Ascend310P3`, `Ascend910A` 等 |
| `--input_format` | `NCHW` 或 `NHWC` |
| `--precision_mode` | `allow_fp32_to_fp16`（默认，精度/性能均衡）|
| `--op_select_implmode` | `high_performance` 或 `high_precision` |
| `--insert_op_conf` | 插入 AIPP 预处理配置文件 |
| `--output_type` | 输出数据类型（FP32/FP16/INT8）|

### 为什么需要多个 .om

310P **不支持动态 batch shape**，必须为每个 batch size 单独编译：

```
models/
├── yolo11n_bs1.om    ← batch=1
├── yolo11n_bs4.om    ← batch=4
├── yolo11n_bs8.om    ← batch=8
└── yolo11n_bs16.om   ← batch=16
```

本项目的 `AscendBackend::selectModel()` 在运行时根据实际 batch 大小选择最接近的 .om，这是标准做法。

### 加载与运行

```cpp
// 加载模型
aclmdlLoadFromFile("yolo11n_bs8.om", &model_id_);
aclmdlGetDesc(model_desc_, model_id_);

// 创建输入/输出数据集
aclmdlDataset* input  = aclmdlCreateDataset();
aclmdlDataset* output = aclmdlCreateDataset();

// 执行推理（异步，提交到 stream）
aclmdlExecuteAsync(model_id_, input, output, stream);
aclrtSynchronizeStream(stream);   // 等待完成

// 卸载
aclmdlUnload(model_id_);
```

---

## 12. AIPP 预处理加速

### 什么是 AIPP

AIPP（AI Pre-Processor）是昇腾特有功能，将图像预处理**固化进模型**，在 NPU 专用硬件上执行，无需 CPU 介入。

### 处理流程对比

```
不使用 AIPP:                    使用 AIPP:
摄像头 NV12                     摄像头 NV12
  │                               │
  ▼ (CPU/CUDA)                    ▼ (NPU 硬件，自动)
YUV→RGB 转换                  ┌─ AIPP ──────────────┐
resize                        │  YUV→RGB             │
归一化 (mean/std)              │  resize              │
NCHW 排列                     │  归一化 (mean/std)    │
  │                            │  NCHW 排列           │
  ▼                            └─────────────────────┘
YOLO 推理                          │
                                   ▼
                               YOLO 推理
```

### AIPP 配置文件示例

```
aipp_op {
    aipp_mode: static

    input_format: YUV420SP_U8       # NV12 输入

    src_image_size_w: 1920          # 原始图像宽
    src_image_size_h: 1080          # 原始图像高

    crop: false
    resize: true                    # 启用 resize
    resize_output_w: 640
    resize_output_h: 640

    mean_chn_0: 0                   # RGB 均值
    mean_chn_1: 0
    mean_chn_2: 0
    min_chn_0: 0.0                  # 归一化: (pixel - min) / var
    min_chn_1: 0.0
    min_chn_2: 0.0
    var_reci_chn_0: 0.00392157      # 1/255
    var_reci_chn_1: 0.00392157
    var_reci_chn_2: 0.00392157
}
```

ATC 转换时加入 `--insert_op_conf=aipp.cfg` 即可。

---

## 13. DVPP → AIPP 零拷贝管道

### 概述

当同时启用 DVPP 硬件解码（`use_ascend_dvpp: true`）和 NV12 AIPP 模型时，推理服务器可以实现**端到端零 CPU 参与**的推理路径：

```
RTSP 码流
  │
  ▼  (FFmpeg 仅 demux，不解码)
DVPP 硬件视频解码
  │ YUV420SP (NV12) 在 HBM 设备内存
  │  Frame.is_ascend = true
  ▼
BatchScheduler / InferEngineWorkerStage
  │ 填入 Batch.ascend_frames（不走 CPU）
  ▼
AscendBackend::infer() — Path A
  │
  ├── batch=1：DVPP device ptr 直传 ACL dataset（无任何拷贝）
  │
  └── batch>1：D2D memcpy 拼接到临时 HBM buffer（仍在 NPU 侧）
         │
         ▼
      AIPP（.om 内嵌）
        NV12 → RGB → resize → 归一化 → NCHW
         │（NPU 硬件执行，无 CPU 参与）
         ▼
      YOLO NPU 推理
         │
         ▼
      Kafka 结果发布
```

### 与原有路径对比

| 路径 | CPU 操作 | 拷贝方向 | 适用场景 |
|------|----------|----------|---------|
| **Path A 零拷贝** | 无 | 无（batch=1）/ D2D（batch>1） | DVPP + NV12 AIPP 模型 |
| Path B CPU BGR | `cv::resize` + `packBgrUint8` | H2D | AIPP 模型 + FFmpeg 软解 |
| Path C CPU float | `cv::resize` + CHW 归一化 | H2D | 无 AIPP 模型 |

路径自动选择，无需代码改动：`is_ascend && aipp_enabled_` 同时为 true 时激活 Path A。

### 模型编译（NV12 AIPP 配置）

AIPP 配置文件 `aipp_nv12.cfg` 示例（1080p 输入，缩放至 640×640）：

```
aipp_op {
    aipp_mode: static

    input_format: YUV420SP_U8       # 接受 DVPP 输出的 NV12 格式

    src_image_size_w: 1920          # 必须与 DVPP 输出分辨率完全一致
    src_image_size_h: 1080

    crop: false
    resize: true
    resize_output_w: 640
    resize_output_h: 640

    csc_switch: true                # YUV → RGB 色域转换
    rbuv_swap_switch: false

    mean_chn_0: 0
    mean_chn_1: 0
    mean_chn_2: 0
    min_chn_0: 0.0
    min_chn_1: 0.0
    min_chn_2: 0.0
    var_reci_chn_0: 0.00392157      # 1/255 归一化
    var_reci_chn_1: 0.00392157
    var_reci_chn_2: 0.00392157
}
```

ATC 转换命令（批量编译 batch=1/4/8/16）：

```bash
for BS in 1 4 8 16; do
    atc \
      --model=yolo11n.onnx \
      --framework=5 \
      --output="yolo11n_nv12_bs${BS}" \
      --input_shape="images:${BS},3,640,640" \
      --input_format=NCHW \
      --soc_version=Ascend310P3 \
      --insert_op_conf=aipp_nv12.cfg \   # 嵌入 NV12 AIPP
      --precision_mode=allow_fp32_to_fp16
done
```

> **注意**：`src_image_size_w/h` 必须与摄像头实际输出分辨率完全匹配，AscendBackend 运行时不校验，不匹配会导致 NPU 内存越界。

### 配置（config.yaml）

```yaml
streams:
  - id: cam01
    url: "rtsp://192.168.1.100/stream"
    use_ascend_dvpp: true     # 启用 DVPP 硬件解码
    ascend_device_id: 0       # 使用 NPU device 0

models:
  - id: yolo_nv12
    type: yolov8
    om_paths:
      1:  models/yolo11n_nv12_bs1.om
      4:  models/yolo11n_nv12_bs4.om
      8:  models/yolo11n_nv12_bs8.om
      16: models/yolo11n_nv12_bs16.om
    # AscendBackend 自动检测 AIPP（aclmdlGetFirstAippInfo），无需额外配置
```

### 注意事项

- batch=1 真零拷贝，适合高分辨率（1080p/4K）低延迟场景
- batch>1 的 D2D 拼接使用动态 `aclrtMalloc`，未来版本将引入 pool 预分配
- DVPP 与 AIPP 必须同时启用才能走 Path A；单独 DVPP（不含 AIPP）不支持零拷贝

---

## 14. 常见错误速查


| 错误信息 | 原因 | 解决方法 |
|---------|------|---------|
| `DrvMngGetConsoleLogLevel failed` | 驱动与 CANN 版本不匹配 | 对齐驱动与 CANN 版本 |
| `ACL_ERROR_RT_CONTEXT_NULL` | Context 未初始化或跨线程使用 | 确认每线程独立创建 Context |
| `ACL_ERROR_INVALID_DEVICE_ID` | device_id 超出范围 | `npu-smi info -l` 确认可用设备数 |
| `EE9999` | NPU 内部通用错误 | 查 `/var/log/npu/slog/` 详细日志 |
| `model input size mismatch` | 实际 batch != .om 编译时的 batch | `selectModel()` 选择正确的 .om |
| `input format mismatch` | 运行时数据格式与 ATC 编译时不一致 | 重新用正确 `--input_format` 编译 |
| `DVPP_ERROR_*` | 视频硬解码器错误 | 检查 DVPP 初始化流程 |
| `out of memory` | NPU 显存不足 | 减小 batch size 或减少并发模型数 |
| `No parser is registered for Op [... ai.onnx::20::*]` | ONNX opset 过新，ATC 未注册该域 | 重导时指定 `opset=12` 或 `11`（或升级 CANN） |
| `dfl/Reshape` / `Shape size` 与元素个数 4× 等比例不符 | 静态 batch ONNX 与多 batch ATC 不匹配 | `dynamic=True` 重导 ONNX |

---

## 15. 日志与调试

### npu-smi 常用命令

```bash
# 查看所有卡概览
npu-smi info

# 列出所有设备
npu-smi info -l

# 实时监控（卡 0）
npu-smi info -t common -i 0 -c 0

# 查看内存使用
npu-smi info -t memory -i 0

# 查看温度/功耗
npu-smi info -t power -i 0
```

### 日志级别控制

```bash
# 启动前设置（0=DEBUG, 1=INFO, 2=WARN, 3=ERROR）
export ASCEND_GLOBAL_LOG_LEVEL=1

# 同时启用 event log
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

### 日志文件位置

```
/var/log/npu/slog/host-0/           ← 主机侧详细日志
/var/log/npu/slog/device-0/         ← NPU 设备侧日志
/var/log/npu/conf/slog/slog.conf    ← 日志配置
```

### Profiling 性能分析

```bash
# 通过环境变量开启 profiling
export PROFILING_MODE=on
export AICPU_PROFILING_MODE=on

# 或使用 msprof 工具
msprof --output=/tmp/prof_output --application="./inference_server"
```

---

## 16. 本项目适配要点

### 多卡并行

一张 Atlas 300I Pro = **4 个独立 NPU 设备**（device 0~3），可分配给 4 个 InferWorker 线程：

```cpp
// config.yaml
infer:
  device_ids: [0, 1, 2, 3]   # 每个 worker 绑定不同 device

// InferWorker 构造
InferWorker(int device_id) : device_id_(device_id) {}

void InferWorker::run() {
    aclrtSetDevice(device_id_);        // 绑定本 worker 的设备
    aclrtCreateContext(&ctx_, device_id_);
    backend_->loadModel(model_path_);
    // ...推理循环
}
```

### .om 文件组织建议

```
config/models/
├── yolo11n/
│   ├── yolo11n_bs1.om
│   ├── yolo11n_bs4.om
│   ├── yolo11n_bs8.om
│   └── yolo11n_bs16.om
└── yolo11s/
    ├── yolo11s_bs1.om
    └── ...
```

### 模型转换脚本（参考 `scripts/convert_ascend.sh`）

```bash
#!/bin/bash
ONNX=$1
NAME=$2
SOC=${3:-Ascend310P3}

for BS in 1 4 8 16; do
  atc \
    --model="$ONNX" \
    --framework=5 \
    --output="${NAME}_bs${BS}" \
    --input_shape="images:${BS},3,640,640" \
    --input_format=NCHW \
    --soc_version="$SOC" \
    --precision_mode=allow_fp32_to_fp16
done
```

### `sink.stream` 硬件编码（可选，CANN 6 + 310P）

默认 `sink.stream` 仍使用 **OpenCV 画框 + raw BGR → ffmpeg + libx264** 软编。可选开启 **昇腾 VENC（H.264 ES）+ ffmpeg 仅封装**（`-f h264` 从 stdin 读 Annex-B，`-c:v copy` 推 RTSP/RTMP），减轻 CPU 编码负载。

YAML `with` 字段：

| 键 | 说明 |
|----|------|
| `encoder` | `ffmpeg_x264`（默认）或 `ascend_venc` |
| `ascend_device_id` | 可选；**≥0** 时绑定该 NPU；**省略或 -1** 时使用任务级 `ingest_ascend_device_id`（与 ingest / source 侧 DVPP 等设备上下文一致） |

约束与取舍：

- 需要 **`BUILD_ASCEND_BACKEND=ON`** 且运行环境具备对应 **CANN 6 `aclvenc*`** 与驱动；否则配置 `ascend_venc` 会在阶段构造时失败。
- 输入在 CPU 上由 BGR 转为 **NV12**。VENC 路径将宽高 **向上对齐到 16 的倍数**（H.264 宏块栅格；例如 7680×1144 → 7680×1152），`packSingleBgrToNv12Contiguous` 会按对齐后的尺寸缩放，略有竖直拉伸。**首期建议固定分辨率**；若芯片/文档不支持目标宽高，应退回 `ffmpeg_x264` 或调整分辨率。
- 与推理共用进程时，ACL 由 **`AscendProcessRuntime`** 进程级引用计数管理；纯推流任务也会在 writer `open()` 时 acquire ACL。

### 环境检查清单

在运行容器前，宿主机确认：

```bash
# 1. 驱动已加载
lsmod | grep drv_pcie_host

# 2. 设备节点存在
ls /dev/davinci*

# 3. NPU 卡正常
npu-smi info

# 4. 驱动版本
cat /usr/local/Ascend/driver/version.info
```

在容器内确认：

```bash
# 1. 设备节点已挂载
ls /dev/davinci*

# 2. CANN 库可访问
ls /usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so

# 3. ATC 可用（模型转换机器）
atc --version
```

---

## 参考资源

- [昇腾社区官网](https://www.hiascend.com)
- [CANN 开发文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha002/quickstart/quickstart/quickstart_18_0001.html)
- [AscendCL API 参考](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha002/infacldevg/aclcppdevg/aclcppdevg_000001.html)
- [npu-smi 工具指南](https://www.hiascend.com/document)
- [Atlas 300I Pro 产品手册](https://www.hiascend.com/hardware/product)
