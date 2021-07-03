cmake_minimum_required(VERSION 3.19)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(addernet CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)

add_executable(addernet-texture CudaRuntime1/kernel-addernet-texture.cu)

set_target_properties(
        addernet-texture
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(addernet-constant CudaRuntime1/kernel-addernet-constant.cu)

set_target_properties(
        addernet-constant
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(addernet-global CudaRuntime1/kernel-addernet-global.cu)

set_target_properties(
        addernet-global
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(addernet-cpu CudaRuntime1/addernet_cpu.cpp)
