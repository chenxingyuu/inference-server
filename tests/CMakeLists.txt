cmake_minimum_required(VERSION 3.24)

# ── GoogleTest via FetchContent ────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ── Lightweight testable library (no GPU/Kafka/FFmpeg/Stream) ──────────────
# Only includes sources that can be tested without hardware dependencies.
add_library(infer_testable STATIC
    ${CMAKE_SOURCE_DIR}/src/common/Config.cpp
    ${CMAKE_SOURCE_DIR}/src/common/Logger.cpp
    ${CMAKE_SOURCE_DIR}/src/decoder/IYOLODecoder.cpp
    ${CMAKE_SOURCE_DIR}/src/decoder/YOLOv5Decoder.cpp
    ${CMAKE_SOURCE_DIR}/src/decoder/YOLOv8Decoder.cpp
    ${CMAKE_SOURCE_DIR}/src/decoder/YOLO11Decoder.cpp
    ${CMAKE_SOURCE_DIR}/src/decoder/YOLO26Decoder.cpp
    ${CMAKE_SOURCE_DIR}/src/decoder/ClassifierDecoder.cpp
    ${CMAKE_SOURCE_DIR}/src/decoder/DecoderFactory.cpp
    ${CMAKE_SOURCE_DIR}/src/tracker/ByteTrackTracker.cpp
    ${CMAKE_SOURCE_DIR}/src/tracker/TrackerManager.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/ResultMerger.cpp
    # Phase 10: no hardware dependencies
    ${CMAKE_SOURCE_DIR}/src/stream/StreamHealthRegistry.cpp
    ${CMAKE_SOURCE_DIR}/src/stream/ReconnectPolicy.cpp
    ${CMAKE_SOURCE_DIR}/src/stream/GpuDeviceAllocator.cpp
    ${CMAKE_SOURCE_DIR}/src/stream/StopAwareSleep.cpp
    ${CMAKE_SOURCE_DIR}/src/metrics/Metrics.cpp
)

target_include_directories(infer_testable PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${Boost_INCLUDE_DIRS}
)

target_link_libraries(infer_testable PUBLIC
    Threads::Threads
    yaml-cpp
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
)

# ── Helper macro ───────────────────────────────────────────────────────────
function(add_infer_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE
        infer_testable
        GTest::gtest_main
        GTest::gmock
    )
    add_test(NAME ${name} COMMAND ${name}
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endfunction()

# ── Test executables ───────────────────────────────────────────────────────
add_infer_test(test_types)
add_infer_test(test_config)
add_infer_test(test_gpu_device_allocator)
add_infer_test(test_iou_nms)
add_infer_test(test_yolov5_decoder)
add_infer_test(test_result_merger)
add_infer_test(test_tracker)
add_infer_test(test_logging_defects)
add_infer_test(test_debug_logging)

add_infer_test(test_read_timeout)
add_infer_test(test_file_source_config)
add_infer_test(test_ffmpeg_log_level)
add_infer_test(test_stop_aware_sleep)

# Phase 10: stream health state machine (no hardware deps)
add_infer_test(test_stream_health)
add_infer_test(test_reconnect_backoff)
add_infer_test(test_reconnect_policy)
add_infer_test(test_sink_ffplay_metrics)
add_infer_test(test_infer_batch_size_metrics)
add_infer_test(test_queue_latency_metrics)
add_executable(test_stage_factory test_stage_factory.cpp)
target_link_libraries(test_stage_factory PRIVATE
    infer_core
    GTest::gtest_main
)
add_test(NAME test_stage_factory COMMAND test_stage_factory
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

add_executable(test_infer_engine_worker_stage test_infer_engine_worker_stage.cpp)
target_link_libraries(test_infer_engine_worker_stage PRIVATE
    infer_core
    GTest::gtest_main
)
add_test(NAME test_infer_engine_worker_stage COMMAND test_infer_engine_worker_stage
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

add_executable(test_ffmpeg_file_decoder test_ffmpeg_file_decoder.cpp)
target_link_libraries(test_ffmpeg_file_decoder PRIVATE
    infer_core
    GTest::gtest_main
)
add_test(NAME test_ffmpeg_file_decoder COMMAND test_ffmpeg_file_decoder
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

add_executable(test_frame_sampling test_frame_sampling.cpp)
target_link_libraries(test_frame_sampling PRIVATE
    infer_core
    GTest::gtest_main
)
add_test(NAME test_frame_sampling COMMAND test_frame_sampling
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_executable(test_draw_and_stream_stage test_draw_and_stream_stage.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/stages/DrawAndStreamStage.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/stages/DetectionOverlay.cpp
)
target_link_libraries(test_draw_and_stream_stage PRIVATE
    infer_testable
    GTest::gtest_main
)
add_test(NAME test_draw_and_stream_stage COMMAND test_draw_and_stream_stage
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

add_executable(test_sahi_scheduler test_sahi_scheduler.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/stages/SahiSchedulerStage.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/stages/SahiMergeStage.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/stages/SahiRoiRegistry.cpp
)
target_link_libraries(test_sahi_scheduler PRIVATE
    infer_testable
    GTest::gtest_main
)
add_test(NAME test_sahi_scheduler COMMAND test_sahi_scheduler
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

add_executable(test_bgr_to_nv12 test_bgr_to_nv12.cpp
    ${CMAKE_SOURCE_DIR}/src/infer/BgrToNv12.cpp
)
target_link_libraries(test_bgr_to_nv12 PRIVATE
    infer_testable
    GTest::gtest_main
)
add_test(NAME test_bgr_to_nv12 COMMAND test_bgr_to_nv12
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

add_executable(test_archive_raw_stage test_archive_raw_stage.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/stages/ArchiveRawStage.cpp
    ${CMAKE_SOURCE_DIR}/src/archive/FrameArchiver.cpp
)
target_include_directories(test_archive_raw_stage PRIVATE
    ${FFMPEG_INCLUDE_DIRS}
)
target_link_directories(test_archive_raw_stage PRIVATE
    ${FFMPEG_LIBRARY_DIRS}
)
target_link_libraries(test_archive_raw_stage PRIVATE
    infer_testable
    GTest::gtest_main
    CURL::libcurl
    ${FFMPEG_LIBRARIES}
)
add_test(NAME test_archive_raw_stage COMMAND test_archive_raw_stage
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

# Phase 10: heartbeat JSON serialization (needs rdkafka for linking HeartbeatPublisher.cpp)
add_executable(test_heartbeat_publisher test_heartbeat_publisher.cpp
    ${CMAKE_SOURCE_DIR}/src/publisher/HeartbeatPublisher.cpp
    ${CMAKE_SOURCE_DIR}/src/stream/StreamHealthRegistry.cpp
    ${CMAKE_SOURCE_DIR}/src/common/Logger.cpp
    ${CMAKE_SOURCE_DIR}/src/metrics/Metrics.cpp
)
target_include_directories(test_heartbeat_publisher PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${RDKAFKA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_directories(test_heartbeat_publisher PRIVATE
    ${RDKAFKA_LIBRARY_DIRS}
)
target_link_libraries(test_heartbeat_publisher PRIVATE
    GTest::gtest_main
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    ${RDKAFKA_LIBRARIES}
    ${OpenCV_LIBS}
    Threads::Threads
)
add_test(NAME test_heartbeat_publisher COMMAND test_heartbeat_publisher
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

# Phase 14: control topic JSON serialization (needs rdkafka for linking ControlPublisher.cpp)
add_executable(test_control_publisher test_control_publisher.cpp
    ${CMAKE_SOURCE_DIR}/src/publisher/ControlPublisher.cpp
    ${CMAKE_SOURCE_DIR}/src/stream/StreamHealthRegistry.cpp
    ${CMAKE_SOURCE_DIR}/src/common/Logger.cpp
)
target_include_directories(test_control_publisher PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${RDKAFKA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_directories(test_control_publisher PRIVATE
    ${RDKAFKA_LIBRARY_DIRS}
)
target_link_libraries(test_control_publisher PRIVATE
    GTest::gtest_main
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    ${RDKAFKA_LIBRARIES}
    ${OpenCV_LIBS}
    Threads::Threads
)
add_test(NAME test_control_publisher COMMAND test_control_publisher
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

# Phase 11: GPU fault self-healing (BatchPolicy, GpuFault classification, Metrics, InferWorker recovery)
# Uses InferWorker directly so it needs its transitive deps that are not in infer_testable.
add_executable(test_gpu_fault_recovery test_gpu_fault_recovery.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/InferWorker.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/InferWorkerGroup.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/CascadeRouter.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/stages/DetectionOverlay.cpp
    ${CMAKE_SOURCE_DIR}/src/archive/FrameArchiver.cpp
)
target_link_libraries(test_gpu_fault_recovery PRIVATE
    infer_testable
    GTest::gtest_main
    GTest::gmock
    CURL::libcurl
)
add_test(NAME test_gpu_fault_recovery COMMAND test_gpu_fault_recovery
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

# Unix socket management server (no GPU/Kafka/FFmpeg deps)
add_executable(test_unix_socket_server test_unix_socket_server.cpp
    ${CMAKE_SOURCE_DIR}/src/server/UnixSocketServer.cpp
    ${CMAKE_SOURCE_DIR}/src/metrics/Metrics.cpp
)
target_link_libraries(test_unix_socket_server PRIVATE
    infer_testable
    GTest::gtest_main
)
add_test(NAME test_unix_socket_server COMMAND test_unix_socket_server
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

add_executable(test_task_manager_placeholder test_task_manager_placeholder.cpp)
target_link_libraries(test_task_manager_placeholder PRIVATE
    infer_core
    GTest::gtest_main
)
add_test(NAME test_task_manager_placeholder COMMAND test_task_manager_placeholder
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

# TrtUtils pure-logic unit tests — no GPU required, no TRT dependency
add_infer_test(test_trt_utils)

# Multi-publisher fan-out (no Kafka/GPU deps)
add_executable(test_multi_publisher test_multi_publisher.cpp
    ${CMAKE_SOURCE_DIR}/src/publisher/MultiPublisher.cpp
    ${CMAKE_SOURCE_DIR}/src/common/Logger.cpp
)
target_include_directories(test_multi_publisher PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(test_multi_publisher PRIVATE
    GTest::gtest_main
    GTest::gmock
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
    Threads::Threads
)
add_test(NAME test_multi_publisher COMMAND test_multi_publisher
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

# Redis Streams publisher (mock client — no real Redis required for tests)
add_executable(test_redis_publisher test_redis_publisher.cpp
    ${CMAKE_SOURCE_DIR}/src/publisher/RedisPublisher.cpp
    ${CMAKE_SOURCE_DIR}/src/common/Logger.cpp
)
target_include_directories(test_redis_publisher PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(test_redis_publisher PRIVATE
    GTest::gtest_main
    GTest::gmock
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
    Threads::Threads
)
add_test(NAME test_redis_publisher COMMAND test_redis_publisher
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

# ── Ascend NPU backend tests (no real NPU required) ────────────────────────
# All tests use initForTest() + injected stubs to avoid real ACL/DVPP hardware.
if(BUILD_ASCEND_BACKEND)
    # Feature 1: AscendBufferPool
    add_executable(test_ascend_buffer_pool
        test_ascend_buffer_pool.cpp
        ${CMAKE_SOURCE_DIR}/src/infer/AscendBufferPool.cpp
        ${CMAKE_SOURCE_DIR}/src/common/Logger.cpp
        ${CMAKE_SOURCE_DIR}/src/metrics/Metrics.cpp
    )
    target_include_directories(test_ascend_buffer_pool PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
    )
    target_compile_definitions(test_ascend_buffer_pool PRIVATE BUILD_ASCEND_BACKEND)
    target_link_libraries(test_ascend_buffer_pool PRIVATE
        GTest::gtest_main
        GTest::gmock
        spdlog::spdlog
        ${OpenCV_LIBS}
        Threads::Threads
        AscendCL::ascendcl
    )
    add_test(NAME test_ascend_buffer_pool COMMAND test_ascend_buffer_pool
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

    # Feature 2 + 3: async inference + AIPP (both in AscendBackend)
    add_executable(test_ascend_async_infer
        test_ascend_async_infer.cpp
        ${CMAKE_SOURCE_DIR}/src/infer/AscendBackend.cpp
        ${CMAKE_SOURCE_DIR}/src/infer/AscendBufferPool.cpp
        ${CMAKE_SOURCE_DIR}/src/common/Logger.cpp
        ${CMAKE_SOURCE_DIR}/src/metrics/Metrics.cpp
    )
    target_include_directories(test_ascend_async_infer PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
    )
    target_compile_definitions(test_ascend_async_infer PRIVATE BUILD_ASCEND_BACKEND)
    target_link_libraries(test_ascend_async_infer PRIVATE
        GTest::gtest_main
        GTest::gmock
        spdlog::spdlog
        ${OpenCV_LIBS}
        Threads::Threads
        AscendCL::ascendcl
        yaml-cpp
        nlohmann_json::nlohmann_json
    )
    add_test(NAME test_ascend_async_infer COMMAND test_ascend_async_infer
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

    add_executable(test_ascend_aipp
        test_ascend_aipp.cpp
        ${CMAKE_SOURCE_DIR}/src/infer/AscendBackend.cpp
        ${CMAKE_SOURCE_DIR}/src/infer/AscendBufferPool.cpp
        ${CMAKE_SOURCE_DIR}/src/common/Logger.cpp
        ${CMAKE_SOURCE_DIR}/src/metrics/Metrics.cpp
    )
    target_include_directories(test_ascend_aipp PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
    )
    target_compile_definitions(test_ascend_aipp PRIVATE BUILD_ASCEND_BACKEND)
    target_link_libraries(test_ascend_aipp PRIVATE
        GTest::gtest_main
        GTest::gmock
        spdlog::spdlog
        ${OpenCV_LIBS}
        Threads::Threads
        AscendCL::ascendcl
        yaml-cpp
        nlohmann_json::nlohmann_json
    )
    add_test(NAME test_ascend_aipp COMMAND test_ascend_aipp
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

    # Feature 5: DVPP → AIPP zero-copy pipeline
    add_executable(test_ascend_dvpp_zerocopy
        test_ascend_dvpp_zerocopy.cpp
        ${CMAKE_SOURCE_DIR}/src/infer/AscendBackend.cpp
        ${CMAKE_SOURCE_DIR}/src/infer/AscendBufferPool.cpp
        ${CMAKE_SOURCE_DIR}/src/common/Logger.cpp
        ${CMAKE_SOURCE_DIR}/src/metrics/Metrics.cpp
    )
    target_include_directories(test_ascend_dvpp_zerocopy PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
    )
    target_compile_definitions(test_ascend_dvpp_zerocopy PRIVATE BUILD_ASCEND_BACKEND)
    target_link_libraries(test_ascend_dvpp_zerocopy PRIVATE
        GTest::gtest_main
        GTest::gmock
        spdlog::spdlog
        ${OpenCV_LIBS}
        Threads::Threads
        AscendCL::ascendcl
        yaml-cpp
        nlohmann_json::nlohmann_json
    )
    add_test(NAME test_ascend_dvpp_zerocopy COMMAND test_ascend_dvpp_zerocopy
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

    # Feature 4: DVPPDecoder
    add_executable(test_dvpp_decoder
        test_dvpp_decoder.cpp
        ${CMAKE_SOURCE_DIR}/src/stream/DVPPDecoder.cpp
        ${CMAKE_SOURCE_DIR}/src/common/Logger.cpp
        ${CMAKE_SOURCE_DIR}/src/metrics/Metrics.cpp
    )
    target_include_directories(test_dvpp_decoder PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${FFMPEG_INCLUDE_DIRS}
    )
    target_link_directories(test_dvpp_decoder PRIVATE
        ${FFMPEG_LIBRARY_DIRS}
    )
    target_compile_definitions(test_dvpp_decoder PRIVATE BUILD_ASCEND_BACKEND)
    target_link_libraries(test_dvpp_decoder PRIVATE
        GTest::gtest_main
        GTest::gmock
        spdlog::spdlog
        ${OpenCV_LIBS}
        Threads::Threads
        AscendCL::ascendcl
        $<$<TARGET_EXISTS:AscendCL::dvpp>:AscendCL::dvpp>
        ${FFMPEG_LIBRARIES}
        yaml-cpp
        nlohmann_json::nlohmann_json
    )
    add_test(NAME test_dvpp_decoder COMMAND test_dvpp_decoder
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif() # BUILD_ASCEND_BACKEND

# gRPC publisher (requires BUILD_GRPC_PUBLISHER=ON)
if(BUILD_GRPC_PUBLISHER)
    add_executable(test_grpc_publisher test_grpc_publisher.cpp
        ${CMAKE_SOURCE_DIR}/src/publisher/GrpcPublisher.cpp
        ${CMAKE_SOURCE_DIR}/src/common/Logger.cpp
    )
    target_include_directories(test_grpc_publisher PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/generated
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(test_grpc_publisher PRIVATE
        GTest::gtest_main
        GTest::gmock
        inference_proto
        gRPC::grpc++
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        ${OpenCV_LIBS}
        Threads::Threads
    )
    add_test(NAME test_grpc_publisher COMMAND test_grpc_publisher
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
