# FindTensorRT.cmake
# Locates TensorRT 10.x installation.
# Sets: TensorRT_FOUND, TensorRT::nvinfer

find_path(TensorRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS
        $ENV{TRT_ROOT}/include
        /usr/include/x86_64-linux-gnu
        /usr/local/include
        /usr/include
)

find_library(TensorRT_LIBRARY
    NAMES nvinfer
    PATHS
        $ENV{TRT_ROOT}/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
)

find_library(TensorRT_ONNX_LIBRARY
    NAMES nvonnxparser
    PATHS
        $ENV{TRT_ROOT}/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TensorRT_LIBRARY TensorRT_INCLUDE_DIR)

if(TensorRT_FOUND AND NOT TARGET TensorRT::nvinfer)
    add_library(TensorRT::nvinfer SHARED IMPORTED)
    set_target_properties(TensorRT::nvinfer PROPERTIES
        IMPORTED_LOCATION             "${TensorRT_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
    )

    if(TensorRT_ONNX_LIBRARY)
        add_library(TensorRT::nvonnxparser SHARED IMPORTED)
        set_target_properties(TensorRT::nvonnxparser PROPERTIES
            IMPORTED_LOCATION             "${TensorRT_ONNX_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
        )
    endif()
endif()
