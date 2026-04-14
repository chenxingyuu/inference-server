cmake_minimum_required(VERSION 3.20)
project(InferenceServer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Build options ──────────────────────────────────────────────────────────────
option(BUILD_TRT_BACKEND    "Build TensorRT backend"    ON)
option(BUILD_ASCEND_BACKEND "Build Ascend ACL backend"  OFF)
option(BUILD_TESTS          "Build unit tests"           OFF)
option(ENABLE_ASAN          "Enable AddressSanitizer"   OFF)

# ── Module path ───────────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# ── Compiler flags ────────────────────────────────────────────────────────────
include(CompilerOptions)

# ── Required packages ─────────────────────────────────────────────────────────
find_package(Threads REQUIRED)

# FFmpeg
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale)

# yaml-cpp
find_package(yaml-cpp REQUIRED)

# nlohmann/json
find_package(nlohmann_json 3.9 REQUIRED)

# spdlog
find_package(spdlog REQUIRED)

# Boost (lockfree queues)
find_package(Boost 1.71 REQUIRED)

# OpenCV
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc)

# librdkafka
pkg_check_modules(RDKAFKA REQUIRED rdkafka)

# ── Optional backends ─────────────────────────────────────────────────────────
if(BUILD_TRT_BACKEND)
    find_package(TensorRT REQUIRED)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
endif()

if(BUILD_ASCEND_BACKEND)
    find_package(AscendCL REQUIRED)
endif()

# ── Source files ──────────────────────────────────────────────────────────────
set(COMMON_SOURCES
    src/common/Config.cpp
    src/common/Logger.cpp
)

set(STREAM_SOURCES
    src/stream/FFmpegDecoder.cpp
    src/stream/FrameBuffer.cpp
    src/stream/StreamPool.cpp
)

set(INFER_SOURCES
    src/infer/BackendFactory.cpp
)
if(BUILD_TRT_BACKEND)
    list(APPEND INFER_SOURCES src/infer/TRTBackend.cpp)
endif()
if(BUILD_ASCEND_BACKEND)
    list(APPEND INFER_SOURCES src/infer/AscendBackend.cpp)
endif()

set(DECODER_SOURCES
    src/decoder/YOLOv5Decoder.cpp
    src/decoder/YOLOv8Decoder.cpp
    src/decoder/YOLO11Decoder.cpp
    src/decoder/YOLO26Decoder.cpp
    src/decoder/DecoderFactory.cpp
)

set(PIPELINE_SOURCES
    src/pipeline/BatchScheduler.cpp
    src/pipeline/InferWorker.cpp
)

set(PUBLISHER_SOURCES
    src/publisher/KafkaPublisher.cpp
)

# ── Library ───────────────────────────────────────────────────────────────────
add_library(infer_core STATIC
    ${COMMON_SOURCES}
    ${STREAM_SOURCES}
    ${INFER_SOURCES}
    ${DECODER_SOURCES}
    ${PIPELINE_SOURCES}
    ${PUBLISHER_SOURCES}
)

target_include_directories(infer_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${FFMPEG_INCLUDE_DIRS}
    ${RDKAFKA_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

target_link_libraries(infer_core PUBLIC
    Threads::Threads
    ${FFMPEG_LIBRARIES}
    yaml-cpp
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    ${OpenCV_LIBS}
    ${RDKAFKA_LIBRARIES}
    $<$<BOOL:${BUILD_TRT_BACKEND}>:TensorRT::nvinfer CUDA::cudart>
    $<$<BOOL:${BUILD_ASCEND_BACKEND}>:AscendCL::ascendcl>
)

target_compile_definitions(infer_core PUBLIC
    $<$<BOOL:${BUILD_TRT_BACKEND}>:BUILD_TRT_BACKEND>
    $<$<BOOL:${BUILD_ASCEND_BACKEND}>:BUILD_ASCEND_BACKEND>
)

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(infer_server src/main.cpp)
target_link_libraries(infer_server PRIVATE infer_core)

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(tests)
endif()
