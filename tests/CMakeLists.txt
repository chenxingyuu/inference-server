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
