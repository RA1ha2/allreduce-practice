cmake_minimum_required(VERSION 3.18)
project(RingAllReduce LANGUAGES CXX CUDA)

add_executable(ring_test_multi
    benchmarks/ring_test_multi.cu
    src/allreduce_ring.cu
    src/kernels/elementwise_add.cu
)
target_include_directories(ring_test_multi PRIVATE include)