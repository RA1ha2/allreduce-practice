#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cstring>
#include "allreduce_ring.h"

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

// 运行一个策略的测试
void run_strategy_test(int num_gpus, const char* strategy, int seg_count, int block_size) {
    int test_sizes[] = {
        1 * 1024 * 1024,
        4 * 1024 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
        128 * 1024 * 1024
    };
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    const int warmup = 2, iters = 5;

    printf("\n--- Strategy: %s ---\n", strategy);
    printf("%-12s %15s %15s %15s\n", "Size (MB)", "Total (ms)", "Throughput (GB/s)", "");
    printf("%-12s %15s %15s %15s\n", "-----------", "---------------", "------------------", "");

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

        double total_time_s = 0.0;
        for (int run = 0; run < warmup + iters; ++run) {
            // 重置数据
            for (int i = 0; i < num_gpus; ++i) {
                CUDA_CHECK(cudaSetDevice(i));
                CUDA_CHECK(cudaMemcpy(d_data[i], h_data[i], bytes, cudaMemcpyHostToDevice));
            }
            for (int i = 0; i < num_gpus; ++i) { CUDA_CHECK(cudaSetDevice(i)); CUDA_CHECK(cudaDeviceSynchronize()); }

            auto start = std::chrono::high_resolution_clock::now();

            if (strcmp(strategy, "uniform") == 0) {
                // 统一分区：所有 GPU 一起参与 AllReduce
                for (int i = 0; i < num_gpus; ++i) {
                    CUDA_CHECK(cudaSetDevice(i));
                    ring_reduce_scatter(d_data[i], N, i, num_gpus, d_data);
                }
                for (int i = 0; i < num_gpus; ++i) { CUDA_CHECK(cudaSetDevice(i)); CUDA_CHECK(cudaDeviceSynchronize()); }
                for (int i = 0; i < num_gpus; ++i) {
                    CUDA_CHECK(cudaSetDevice(i));
                    ring_allgather(d_data[i], N, i, num_gpus, d_data);
                }
                for (int i = 0; i < num_gpus; ++i) { CUDA_CHECK(cudaSetDevice(i)); CUDA_CHECK(cudaDeviceSynchronize()); }
            }
            else if (strcmp(strategy, "segmented") == 0) {
                // 分段策略：将数据分成 seg_count 段，逐段执行 AllReduce
                int seg_size = N / seg_count;
                int remainder = N % seg_count;
                int offset = 0;
                for (int seg = 0; seg < seg_count; ++seg) {
                    int cur_size = seg_size + (seg < remainder ? 1 : 0);
                    if (cur_size == 0) continue;
                    for (int i = 0; i < num_gpus; ++i) {
                        CUDA_CHECK(cudaSetDevice(i));
                        ring_reduce_scatter(d_data[i] + offset, cur_size, i, num_gpus, d_data);
                    }
                    for (int i = 0; i < num_gpus; ++i) { CUDA_CHECK(cudaSetDevice(i)); CUDA_CHECK(cudaDeviceSynchronize()); }
                    for (int i = 0; i < num_gpus; ++i) {
                        CUDA_CHECK(cudaSetDevice(i));
                        ring_allgather(d_data[i] + offset, cur_size, i, num_gpus, d_data);
                    }
                    for (int i = 0; i < num_gpus; ++i) { CUDA_CHECK(cudaSetDevice(i)); CUDA_CHECK(cudaDeviceSynchronize()); }
                    offset += cur_size;
                }
            }
            else if (strcmp(strategy, "blockcyclic") == 0) {
                // Block-Cyclic：分成两个子组，组内执行 AllReduce
                int num_subgroups = 2;
                int num_blocks = (N + block_size - 1) / block_size;
                for (int b = 0; b < num_blocks; ++b) {
                    int subgroup_id = b % num_subgroups;
                    // 确定当前块在数据中的范围
                    int block_start = b * block_size;
                    int block_end = (b + 1) * block_size;
                    if (block_end > N) block_end = N;
                    int cur_size = block_end - block_start;

                    // 为当前子组构建设备指针数组（只包含属于该子组的 GPU）
                    float** sub_data = (float**)malloc(num_gpus * sizeof(float*)); // 最多 num_gpus
                    int sub_ranks[8]; // 临时存储子组内的原始 rank
                    int sub_count = 0;
                    for (int i = 0; i < num_gpus; ++i) {
                        if (i % 2 == subgroup_id) { // 简单交替划分：0,2 组0；1,3 组1
                            sub_data[sub_count] = d_data[i];
                            sub_ranks[sub_count] = i;
                            sub_count++;
                        }
                    }
                    if (sub_count <= 1) { // 子组只有一张卡，无需通信，跳过
                        free(sub_data);
                        continue;
                    }

                    // 执行该子组的 AllReduce（注意块内的偏移）
                    float** block_ptrs = (float**)malloc(sub_count * sizeof(float*));
                    for (int i = 0; i < sub_count; ++i) {
                        block_ptrs[i] = sub_data[i] + block_start;
                    }

                    // 子组内的 rank 重映射
                    for (int i = 0; i < sub_count; ++i) {
                        CUDA_CHECK(cudaSetDevice(sub_ranks[i]));
                        ring_reduce_scatter(block_ptrs[i], cur_size, i, sub_count, block_ptrs);
                    }
                    for (int i = 0; i < sub_count; ++i) { CUDA_CHECK(cudaSetDevice(sub_ranks[i])); CUDA_CHECK(cudaDeviceSynchronize()); }
                    for (int i = 0; i < sub_count; ++i) {
                        CUDA_CHECK(cudaSetDevice(sub_ranks[i]));
                        ring_allgather(block_ptrs[i], cur_size, i, sub_count, block_ptrs);
                    }
                    for (int i = 0; i < sub_count; ++i) { CUDA_CHECK(cudaSetDevice(sub_ranks[i])); CUDA_CHECK(cudaDeviceSynchronize()); }

                    free(block_ptrs);
                    free(sub_data);
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            if (run >= warmup) {
                std::chrono::duration<double> elapsed = end - start;
                total_time_s += elapsed.count();
            }
        }

        double avg_time_s = total_time_s / iters;
        double avg_time_ms = avg_time_s * 1000.0;
        double throughput = (double)bytes / avg_time_s / 1e9; // GB/s

        printf("%-12.2f %15.3f %15.2f %15s\n", size_mb, avg_time_ms, throughput, "");
        for (int i = 0; i < num_gpus; ++i) { CUDA_CHECK(cudaSetDevice(i)); CUDA_CHECK(cudaFree(d_data[i])); free(h_data[i]); }
        free(h_data); free(d_data);
    }
}

int main(int argc, char** argv) {
    int num_gpus = 4;
    const char* strategy = "uniform";
    int seg_count = 4;      // segmented 的段数
    int block_size = 512 * 1024; // block-cyclic 的块大小

    if (argc > 1) num_gpus = atoi(argv[1]);
    if (argc > 2) strategy = argv[2];
    if (argc > 3) seg_count = atoi(argv[3]);
    if (argc > 4) block_size = atoi(argv[4]);

    if (num_gpus < 2) { printf("Need at least 2 GPUs.\n"); return 1; }

    int dev_count;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (num_gpus > dev_count) {
        printf("Requested %d GPUs but only %d available. Using %d.\n", num_gpus, dev_count, dev_count);
        num_gpus = dev_count;
    }

    enable_all_p2p(num_gpus);
    printf("Partition Strategy Benchmark on %d GPUs\n", num_gpus);

    run_strategy_test(num_gpus, strategy, seg_count, block_size);

    printf("Benchmark complete.\n");
    return 0;
}
