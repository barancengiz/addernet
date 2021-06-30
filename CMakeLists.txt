cmake_minimum_required(VERSION 3.19)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(assignment1 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(assignment1 CudaRuntime1/kernel.cu)

set_target_properties(
        assignment1
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)