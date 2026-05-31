cmake_minimum_required(VERSION 3.18)
project(RingAllReduce LANGUAGES CXX CUDA)

add_executable(ring_test_multi
    benchmarks/ring_test_multi.cu
    src/allreduce_ring.cu
)

add_executable(ring_benchmark
    benchmarks/ring_benchmark.cu
    src/allreduce_ring.cu
)

add_executable(ring_benchmark_async
    benchmarks/ring_benchmark_async.cu
    src/allreduce_ring.cu
    src/allreduce_ring_async.cu
)

target_include_directories(ring_test_multi PRIVATE include)
target_include_directories(ring_benchmark PRIVATE include)
target_include_directories(ring_benchmark_async PRIVATE include)
