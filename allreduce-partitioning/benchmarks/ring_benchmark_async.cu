#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include "allreduce_ring.h"          // 提供 enable_all_p2p
#include "allreduce_ring_async.h"   // 提供异步函数和 StreamBuffers

#define CUDA_CHECK(call) do {                               \
    cudaError_t err = call;                                 \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA error %s:%d: %s\n",          \
                __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while(0)

void fill_random(float* data, int N) {
    for (int i = 0; i < N; ++i)
        data[i] = static_cast<float>(rand()) / RAND_MAX;
}

int main(int argc, char** argv) {
    int num_gpus = 4;
    if (argc > 1) num_gpus = atoi(argv[1]);
    if (num_gpus < 2) { printf("Need at least 2 GPUs.\n"); return 1; }

    int dev_count;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (num_gpus > dev_count) {
        printf("Requested %d GPUs but only %d available. Using %d.\n", num_gpus, dev_count, dev_count);
        num_gpus = dev_count;
    }

    enable_all_p2p(num_gpus);
    printf("Async Segmented Benchmark on %d GPUs (2-stream overlap)\n", num_gpus);

    int test_sizes[] = {4*1024*1024, 16*1024*1024, 64*1024*1024, 128*1024*1024};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    const int warmup = 2, iters = 5;

    printf("%-12s %15s %15s\n", "Size (MB)", "Total (ms)", "Throughput (GB/s)");
    printf("%-12s %15s %15s\n", "-----------", "---------------", "------------------");

    for (int s = 0; s < num_sizes; ++s) {
        int N = test_sizes[s];
        N = (N / num_gpus) * num_gpus;
        if (N <= 0) continue;
        const size_t bytes = N * sizeof(float);
        double size_mb = bytes / (1024.0 * 1024.0);

        float** h_data = (float**)malloc(num_gpus * sizeof(float*));
        float** d_data = (float**)malloc(num_gpus * sizeof(float*));
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaMalloc(&d_data[i], bytes));
            h_data[i] = (float*)malloc(bytes);
            fill_random(h_data[i], N);
            CUDA_CHECK(cudaMemcpy(d_data[i], h_data[i], bytes, cudaMemcpyHostToDevice));
        }

        // 创建两个流和各自的缓冲区
        cudaStream_t stream0, stream1;
        CUDA_CHECK(cudaStreamCreate(&stream0));
        CUDA_CHECK(cudaStreamCreate(&stream1));

        StreamBuffers bufs0, bufs1;
        bufs0.allocated = false;
        bufs1.allocated = false;
        // 预分配最大可能缓冲区
        int max_chunk = N / num_gpus;
        init_stream_buffers(&bufs0, num_gpus, max_chunk);
        init_stream_buffers(&bufs1, num_gpus, max_chunk);

        double total_time_s = 0.0;
        for (int run = 0; run < warmup + iters; ++run) {
            for (int i = 0; i < num_gpus; ++i) {
                CUDA_CHECK(cudaSetDevice(i));
                CUDA_CHECK(cudaMemcpy(d_data[i], h_data[i], bytes, cudaMemcpyHostToDevice));
            }
            for (int i = 0; i < num_gpus; ++i) { CUDA_CHECK(cudaSetDevice(i)); CUDA_CHECK(cudaDeviceSynchronize()); }

            // 将数据分成两半
            int half = N / 2;
            int N0 = half, N1 = N - half;

            auto start = std::chrono::high_resolution_clock::now();

            // 流0：处理前半段
            for (int i = 0; i < num_gpus; ++i) {
                CUDA_CHECK(cudaSetDevice(i));
                ring_reduce_scatter_async(d_data[i], N0, i, num_gpus, d_data, stream0, &bufs0);
                ring_allgather_async(d_data[i], N0, i, num_gpus, d_data, stream0, &bufs0);
            }

            // 流1：处理后半段（同时进行）
            float** d_data_half2 = (float**)malloc(num_gpus * sizeof(float*));
            for (int i = 0; i < num_gpus; ++i) {
                d_data_half2[i] = d_data[i] + half;
            }
            for (int i = 0; i < num_gpus; ++i) {
                CUDA_CHECK(cudaSetDevice(i));
                ring_reduce_scatter_async(d_data_half2[i], N1, i, num_gpus, d_data_half2, stream1, &bufs1);
                ring_allgather_async(d_data_half2[i], N1, i, num_gpus, d_data_half2, stream1, &bufs1);
            }
            free(d_data_half2);

            // 同步两个流
            CUDA_CHECK(cudaStreamSynchronize(stream0));
            CUDA_CHECK(cudaStreamSynchronize(stream1));

            auto end = std::chrono::high_resolution_clock::now();
            if (run >= warmup) {
                std::chrono::duration<double> elapsed = end - start;
                total_time_s += elapsed.count();
            }
        }

        double avg_time_s = total_time_s / iters;
        double avg_time_ms = avg_time_s * 1000.0;
        double throughput = (double)bytes / avg_time_s / 1e9;

        printf("%-12.2f %15.3f %15.2f\n", size_mb, avg_time_ms, throughput);

        // 清理流和缓冲区
        CUDA_CHECK(cudaStreamDestroy(stream0));
        CUDA_CHECK(cudaStreamDestroy(stream1));
        free_stream_buffers(&bufs0);
        free_stream_buffers(&bufs1);
        for (int i = 0; i < num_gpus; ++i) { CUDA_CHECK(cudaSetDevice(i)); CUDA_CHECK(cudaFree(d_data[i])); free(h_data[i]); }
        free(h_data); free(d_data);
    }
    printf("Benchmark complete.\n");
    return 0;
}
