cmake_minimum_required(VERSION 3.24)
project(InferenceServer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Build options ──────────────────────────────────────────────────────────────
option(BUILD_TRT_BACKEND         "Build TensorRT backend"                 ON)
option(BUILD_ASCEND_BACKEND      "Build Ascend ACL backend"               OFF)
option(BUILD_ONNX_BACKEND        "Build ONNX Runtime backend (CPU + MPS)" OFF)
option(BUILD_ONNX_BACKEND_COREML "Enable CoreML EP for MPS (macOS only)"  OFF)
option(BUILD_TESTS               "Build unit tests"                        OFF)
option(ENABLE_ASAN               "Enable AddressSanitizer"                 OFF)
option(BUILD_GRPC_PUBLISHER      "Build gRPC streaming publisher"          OFF)
option(BUILD_REDIS_PUBLISHER     "Build Redis Streams publisher"           OFF)

# ── Module path ───────────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# ── Compiler flags ────────────────────────────────────────────────────────────
include(CompilerOptions)

# ── Required packages ─────────────────────────────────────────────────────────
find_package(Threads REQUIRED)

# FFmpeg
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale libavdevice libavfilter)

# yaml-cpp
find_package(yaml-cpp REQUIRED)

# nlohmann/json — 3.7 minimum; Ubuntu 20.04 ships 3.7.3 which is sufficient.
find_package(nlohmann_json 3.7 REQUIRED)

# spdlog
find_package(spdlog REQUIRED)

# Boost (lockfree queues)
find_package(Boost 1.71 REQUIRED)

# OpenCV
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio)

# librdkafka
pkg_check_modules(RDKAFKA REQUIRED rdkafka)
find_package(CURL REQUIRED)

# ── Optional backends ─────────────────────────────────────────────────────────
if(BUILD_TRT_BACKEND)
    find_package(TensorRT REQUIRED)
    find_package(CUDAToolkit REQUIRED)
    # Enable CUDA language BEFORE any targets reference .cu files.
    # CMAKE_CUDA_STANDARD must be set before enable_language(CUDA).
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    enable_language(CUDA)
endif()

if(BUILD_ASCEND_BACKEND)
    find_package(AscendCL REQUIRED)
endif()

if(BUILD_ONNX_BACKEND)
    # Try system-installed onnxruntime first, then fall back to FetchContent
    find_package(onnxruntime QUIET)
    if(NOT onnxruntime_FOUND)
        include(FetchContent)
        if(APPLE)
            set(_ORT_ARCHIVE "onnxruntime-osx-arm64-1.18.0.tgz")
        elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            set(_ORT_ARCHIVE "onnxruntime-linux-x64-1.18.0.tgz")
        else()
            message(FATAL_ERROR "BUILD_ONNX_BACKEND: unsupported platform for auto-download; install onnxruntime manually")
        endif()
        FetchContent_Declare(onnxruntime_download
            URL "https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/${_ORT_ARCHIVE}"
        )
        FetchContent_MakeAvailable(onnxruntime_download)
        # Create an imported target mirroring the installed package layout
        add_library(onnxruntime SHARED IMPORTED)
        if(APPLE)
            set(_ORT_LIB "${onnxruntime_download_SOURCE_DIR}/lib/libonnxruntime.dylib")
        else()
            set(_ORT_LIB "${onnxruntime_download_SOURCE_DIR}/lib/libonnxruntime.so")
        endif()
        set_target_properties(onnxruntime PROPERTIES
            IMPORTED_LOCATION "${_ORT_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_download_SOURCE_DIR}/include"
        )
    endif()
endif()


# ── Source files ──────────────────────────────────────────────────────────────
set(COMMON_SOURCES
    src/common/Config.cpp
    src/common/Logger.cpp
    src/common/RuntimeState.cpp
)

set(STREAM_SOURCES
    src/stream/FFmpegDecoder.cpp
    src/stream/FrameBuffer.cpp
    src/stream/StreamPool.cpp
    src/stream/StreamHealthRegistry.cpp
    src/stream/GpuDeviceAllocator.cpp
)

set(INFER_SOURCES
    src/infer/BackendFactory.cpp
)
if(BUILD_TRT_BACKEND)
    list(APPEND INFER_SOURCES src/infer/TRTBackend.cpp)
    list(APPEND INFER_SOURCES src/infer/GpuBufferPool.cpp)
    list(APPEND INFER_SOURCES src/cuda/CudaPreprocess.cu)
endif()
if(BUILD_ASCEND_BACKEND)
    list(APPEND INFER_SOURCES src/infer/AscendBackend.cpp)
    list(APPEND INFER_SOURCES src/infer/AscendBufferPool.cpp)
    list(APPEND STREAM_SOURCES src/stream/DVPPDecoder.cpp)
endif()
if(BUILD_ONNX_BACKEND)
    list(APPEND INFER_SOURCES src/infer/OnnxBackend.cpp)
endif()

set(DECODER_SOURCES
    src/decoder/IYOLODecoder.cpp
    src/decoder/YOLOv5Decoder.cpp
    src/decoder/YOLOv8Decoder.cpp
    src/decoder/YOLO11Decoder.cpp
    src/decoder/YOLO26Decoder.cpp
    src/decoder/ClassifierDecoder.cpp
    src/decoder/DecoderFactory.cpp
)

set(PIPELINE_SOURCES
    src/pipeline/EdgeQueue.cpp
    src/pipeline/GraphExecutor.cpp
    src/pipeline/StageFactory.cpp
    src/pipeline/stages/ArchiveRawStage.cpp
    src/pipeline/stages/DetectionOverlay.cpp
    src/pipeline/stages/DrawAndStreamStage.cpp
    src/pipeline/stages/SinkFfplayStage.cpp
    src/pipeline/stages/InferEngineStage.cpp
    src/pipeline/stages/InferEngineWorkerStage.cpp
    src/pipeline/stages/JoinByFrameStage.cpp
    src/pipeline/stages/SinkKafkaStage.cpp
    src/pipeline/stages/SourceFileStage.cpp
    src/pipeline/stages/SourceRtspStage.cpp
    src/pipeline/stages/TrackByteTrackStage.cpp
    src/pipeline/TaskManager.cpp
    src/pipeline/BatchScheduler.cpp
    src/pipeline/InferWorker.cpp
    src/pipeline/InferWorkerGroup.cpp
    src/pipeline/CascadeRouter.cpp
    src/pipeline/ResultMerger.cpp
    src/pipeline/ModelManager.cpp
)

set(TRACKER_SOURCES
    src/tracker/ByteTrackTracker.cpp
    src/tracker/TrackerManager.cpp
)

set(PUBLISHER_SOURCES
    src/publisher/KafkaPublisher.cpp
    src/publisher/AttributePublisher.cpp
    src/publisher/HeartbeatPublisher.cpp
    src/publisher/ControlPublisher.cpp
    src/publisher/ControlEventBus.cpp
    src/publisher/MultiPublisher.cpp
    $<$<BOOL:${BUILD_GRPC_PUBLISHER}>:${CMAKE_SOURCE_DIR}/src/publisher/GrpcPublisher.cpp>
    $<$<BOOL:${BUILD_REDIS_PUBLISHER}>:${CMAKE_SOURCE_DIR}/src/publisher/RedisPublisher.cpp>
)

# ── gRPC publisher ────────────────────────────────────────────────────────────
if(BUILD_GRPC_PUBLISHER)
    find_package(gRPC CONFIG QUIET)
    if(NOT gRPC_FOUND)
        # Fallback: apt-installed gRPC on Ubuntu provides pkg-config but not cmake config
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(GRPCPP REQUIRED grpc++)
        if(NOT TARGET gRPC::grpc++)
            add_library(gRPC::grpc++ INTERFACE IMPORTED)
            set_target_properties(gRPC::grpc++ PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${GRPCPP_INCLUDE_DIRS}"
                INTERFACE_LINK_LIBRARIES "${GRPCPP_LINK_LIBRARIES}"
                INTERFACE_LINK_DIRECTORIES "${GRPCPP_LIBRARY_DIRS}"
                INTERFACE_COMPILE_OPTIONS "${GRPCPP_CFLAGS_OTHER}"
            )
        endif()
        if(NOT TARGET protobuf::libprotobuf)
            pkg_check_modules(PROTOBUF REQUIRED protobuf)
            add_library(protobuf::libprotobuf INTERFACE IMPORTED)
            set_target_properties(protobuf::libprotobuf PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${PROTOBUF_INCLUDE_DIRS}"
                INTERFACE_LINK_LIBRARIES "${PROTOBUF_LINK_LIBRARIES}"
                INTERFACE_LINK_DIRECTORIES "${PROTOBUF_LIBRARY_DIRS}"
            )
        endif()
        message(STATUS "gRPC found via pkg-config: ${GRPCPP_VERSION}")
    else()
        message(STATUS "gRPC found via cmake config: ${gRPC_VERSION}")
    endif()

    set(PROTO_SRC_DIR "${CMAKE_SOURCE_DIR}/proto")
    set(PROTO_OUT_DIR "${CMAKE_BINARY_DIR}/generated")
    file(MAKE_DIRECTORY ${PROTO_OUT_DIR})

    find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)
    # gRPC's protobuf import does not always expose Protobuf_PROTOC_EXECUTABLE to this scope.
    find_program(INFER_PROTOC_EXECUTABLE protoc REQUIRED)

    add_custom_command(
        OUTPUT
            "${PROTO_OUT_DIR}/inference.pb.cc"
            "${PROTO_OUT_DIR}/inference.pb.h"
            "${PROTO_OUT_DIR}/inference.grpc.pb.cc"
            "${PROTO_OUT_DIR}/inference.grpc.pb.h"
        COMMAND ${INFER_PROTOC_EXECUTABLE}
            -I ${PROTO_SRC_DIR}
            --cpp_out=${PROTO_OUT_DIR}
            --grpc_out=${PROTO_OUT_DIR}
            --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
            ${PROTO_SRC_DIR}/inference.proto
        DEPENDS "${PROTO_SRC_DIR}/inference.proto"
    )

    add_library(inference_proto STATIC
        "${PROTO_OUT_DIR}/inference.pb.cc"
        "${PROTO_OUT_DIR}/inference.grpc.pb.cc"
    )
    target_include_directories(inference_proto PUBLIC ${PROTO_OUT_DIR})
    target_link_libraries(inference_proto PUBLIC
        protobuf::libprotobuf
        gRPC::grpc++
    )
endif()

# ── Redis publisher ───────────────────────────────────────────────────────────
# When FetchContent is used, always bundle hiredis too. If only redis++ is missing but
# system hiredis is found, skipping FetchContent hiredis makes redis++ compile against
# Homebrew hiredis headers while redis++ enables -Werror (Clang flexible-array-member).
if(BUILD_REDIS_PUBLISHER)
    find_package(redis++ QUIET)
    if(NOT redis++_FOUND)
        include(FetchContent)
        FetchContent_Declare(hiredis
            GIT_REPOSITORY https://github.com/redis/hiredis.git
            GIT_TAG        v1.3.0
            GIT_SHALLOW    TRUE
            OVERRIDE_FIND_PACKAGE
        )
        set(DISABLE_TESTS ON CACHE BOOL "" FORCE)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(hiredis)

        # redis++ FindHiredis: OVERRIDE_FIND_PACKAGE leaves hiredis_INCLUDE_DIRS empty and its
        # fallback find_path does not match FetchContent's hiredis-src layout. Provide
        # include/hiredis/*.h so <hiredis/hiredis.h> resolves against the bundled tree.
        set(INFER_FETCHED_HIREDIS_INCLUDE "${CMAKE_BINARY_DIR}/hiredis_vendor_include")
        if(EXISTS "${INFER_FETCHED_HIREDIS_INCLUDE}")
            file(REMOVE_RECURSE "${INFER_FETCHED_HIREDIS_INCLUDE}")
        endif()
        file(MAKE_DIRECTORY "${INFER_FETCHED_HIREDIS_INCLUDE}/hiredis")
        file(GLOB _INFER_HIREDIS_HDRS "${hiredis_SOURCE_DIR}/*.h")
        foreach(_hdr IN LISTS _INFER_HIREDIS_HDRS)
            get_filename_component(_hname "${_hdr}" NAME)
            file(CREATE_LINK "${_hdr}" "${INFER_FETCHED_HIREDIS_INCLUDE}/hiredis/${_hname}" SYMBOLIC)
        endforeach()
        if(IS_DIRECTORY "${hiredis_SOURCE_DIR}/adapters")
            file(CREATE_LINK "${hiredis_SOURCE_DIR}/adapters"
                 "${INFER_FETCHED_HIREDIS_INCLUDE}/hiredis/adapters" SYMBOLIC)
        endif()
        set(hiredis_INCLUDE_DIRS "${INFER_FETCHED_HIREDIS_INCLUDE}" CACHE INTERNAL "" FORCE)

        FetchContent_Declare(redis_plus_plus
            GIT_REPOSITORY https://github.com/sewenew/redis-plus-plus.git
            GIT_TAG        1.3.15
            GIT_SHALLOW    TRUE
        )
        set(REDIS_PLUS_PLUS_BUILD_TEST   OFF CACHE BOOL "" FORCE)
        set(REDIS_PLUS_PLUS_BUILD_STATIC ON  CACHE BOOL "" FORCE)
        set(REDIS_PLUS_PLUS_BUILD_SHARED OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(redis_plus_plus)
        # redis++ enables -Werror; Clang rejects hiredis C headers (flexible array members) in C++ mode.
        foreach(_rpp_tgt IN ITEMS redis++_static redis++)
            if(TARGET ${_rpp_tgt})
                target_compile_options(${_rpp_tgt} PRIVATE
                    $<$<COMPILE_LANGUAGE:CXX>:-Wno-error>
                    $<$<COMPILE_LANGUAGE:CXX>:-Wno-c99-extensions>)
            endif()
        endforeach()
    endif()
endif()

set(ARCHIVE_SOURCES
    src/archive/FrameArchiver.cpp
)

set(METRICS_SOURCES
    src/metrics/Metrics.cpp
)

set(SERVER_SOURCES
    src/server/UnixSocketServer.cpp
)

# ── Library ───────────────────────────────────────────────────────────────────
add_library(infer_core STATIC
    ${COMMON_SOURCES}
    ${STREAM_SOURCES}
    ${INFER_SOURCES}
    ${DECODER_SOURCES}
    ${PIPELINE_SOURCES}
    ${TRACKER_SOURCES}
    ${PUBLISHER_SOURCES}
    ${ARCHIVE_SOURCES}
    ${METRICS_SOURCES}
    ${SERVER_SOURCES}
)

target_include_directories(infer_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/cuda
    ${FFMPEG_INCLUDE_DIRS}
    ${RDKAFKA_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    $<$<BOOL:${BUILD_GRPC_PUBLISHER}>:${PROTO_OUT_DIR}>
)

target_link_directories(infer_core PUBLIC
    ${FFMPEG_LIBRARY_DIRS}
    ${RDKAFKA_LIBRARY_DIRS}
)

target_link_libraries(infer_core PUBLIC
    Threads::Threads
    ${FFMPEG_LIBRARIES}
    yaml-cpp
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    ${OpenCV_LIBS}
    ${RDKAFKA_LIBRARIES}
    CURL::libcurl
$<$<BOOL:${BUILD_TRT_BACKEND}>:TensorRT::nvinfer>
    $<$<BOOL:${BUILD_TRT_BACKEND}>:CUDA::cudart>
    $<$<BOOL:${BUILD_ASCEND_BACKEND}>:AscendCL::ascendcl>
    $<$<AND:$<BOOL:${BUILD_ASCEND_BACKEND}>,$<TARGET_EXISTS:AscendCL::dvpp>>:AscendCL::dvpp>
    $<$<BOOL:${BUILD_ONNX_BACKEND}>:onnxruntime>
    $<$<BOOL:${BUILD_GRPC_PUBLISHER}>:inference_proto>
    $<$<BOOL:${BUILD_GRPC_PUBLISHER}>:gRPC::grpc++>
)

# FetchContent / find_package redis++ exposes static and/or shared targets.
if(BUILD_REDIS_PUBLISHER)
    if(TARGET redis++::redis++_static)
        target_link_libraries(infer_core PUBLIC redis++::redis++_static)
    elseif(TARGET redis++::redis++)
        target_link_libraries(infer_core PUBLIC redis++::redis++)
    elseif(TARGET redis++)
        target_link_libraries(infer_core PUBLIC redis++)
    else()
        message(FATAL_ERROR "BUILD_REDIS_PUBLISHER is ON but no redis++ CMake target is defined")
    endif()
endif()

target_compile_definitions(infer_core PUBLIC
    $<$<BOOL:${BUILD_TRT_BACKEND}>:BUILD_TRT_BACKEND>
    $<$<BOOL:${BUILD_ASCEND_BACKEND}>:BUILD_ASCEND_BACKEND>
    $<$<BOOL:${BUILD_ASCEND_BACKEND}>:ENABLE_DVPP_INTERFACE>
    $<$<BOOL:${BUILD_ASCEND_BACKEND}>:CANN_VERSION_MAJOR=${CANN_VERSION_MAJOR}>
    $<$<BOOL:${BUILD_ASCEND_BACKEND}>:CANN_VERSION_MINOR=${CANN_VERSION_MINOR}>
    $<$<BOOL:${BUILD_ONNX_BACKEND}>:BUILD_ONNX_BACKEND>
    $<$<BOOL:${BUILD_ONNX_BACKEND_COREML}>:BUILD_ONNX_BACKEND_COREML>
    $<$<BOOL:${BUILD_GRPC_PUBLISHER}>:BUILD_GRPC_PUBLISHER>
    $<$<BOOL:${BUILD_REDIS_PUBLISHER}>:BUILD_REDIS_PUBLISHER>
)

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(infer_server src/main.cpp)
target_link_libraries(infer_server PRIVATE infer_core)

# AscendCL pulls additional toolkit libs (e.g. libascend_hal.so) through DT_NEEDED.
# In containerized builds, help the linker resolve those indirect deps from CANN lib64.
if(BUILD_ASCEND_BACKEND)
    target_link_options(infer_server PRIVATE
        -Wl,-rpath-link,/usr/local/Ascend/ascend-toolkit/latest/lib64
        -Wl,--allow-shlib-undefined
    )
endif()

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
