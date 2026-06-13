cmake_minimum_required(VERSION 3.20)

project(edgeseg_relu_linear_attention_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

set(TENSORRT_ROOT "E:/NVIDIA/TensorRT-8.6.1.6" CACHE PATH "TensorRT 8.6.1 Windows zip root")

find_package(CUDAToolkit REQUIRED)

add_library(edgeseg_relu_linear_attention_plugin SHARED
    include/edgeseg_relu_linear_attention_plugin.h
    src/edgeseg_relu_linear_attention_plugin.cpp
    src/relu_linear_attention_kernel.cu
)

set_target_properties(edgeseg_relu_linear_attention_plugin PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_definitions(edgeseg_relu_linear_attention_plugin PRIVATE
    NOMINMAX
)

target_include_directories(edgeseg_relu_linear_attention_plugin PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${TENSORRT_ROOT}/include"
    "${CUDAToolkit_INCLUDE_DIRS}"
)

target_link_directories(edgeseg_relu_linear_attention_plugin PRIVATE
    "${TENSORRT_ROOT}/lib"
)

target_link_libraries(edgeseg_relu_linear_attention_plugin PRIVATE
    nvinfer
    nvinfer_plugin
    CUDA::cudart
)

if (MSVC)
    target_compile_options(edgeseg_relu_linear_attention_plugin PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /utf-8>
    )
else()
    target_compile_options(edgeseg_relu_linear_attention_plugin PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    )
endif()
