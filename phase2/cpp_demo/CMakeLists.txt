cmake_minimum_required(VERSION 3.20)

project(edgeseg_trt_runtime_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TENSORRT_ROOT "E:/NVIDIA/TensorRT-8.6.1.6" CACHE PATH "TensorRT 8.6.1 Windows zip root")

find_package(CUDAToolkit REQUIRED)

add_executable(trt_runtime_demo trt_runtime_demo.cpp)

target_include_directories(trt_runtime_demo PRIVATE
    "${TENSORRT_ROOT}/include"
    "${CUDAToolkit_INCLUDE_DIRS}"
)

target_link_directories(trt_runtime_demo PRIVATE
    "${TENSORRT_ROOT}/lib"
)

target_link_libraries(trt_runtime_demo PRIVATE
    nvinfer
    nvinfer_plugin
    CUDA::cudart
)

if (MSVC)
    target_compile_options(trt_runtime_demo PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(trt_runtime_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()
