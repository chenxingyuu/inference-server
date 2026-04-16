cmake_minimum_required(VERSION 3.20)

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
add_infer_test(test_iou_nms)
add_infer_test(test_yolov5_decoder)
add_infer_test(test_result_merger)
add_infer_test(test_tracker)

# Phase 10: stream health state machine (no hardware deps)
add_infer_test(test_stream_health)
add_infer_test(test_reconnect_backoff)

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
