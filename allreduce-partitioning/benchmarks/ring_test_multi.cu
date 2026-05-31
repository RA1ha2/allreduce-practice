#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "allreduce_ring.h"

// 外部声明：内核定义在 allreduce_ring.cu 中
extern __global__ void simple_add_kernel(float* data, const float* recv, int n);

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
    int N = 1024 * 1024;
    if (argc > 1) num_gpus = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (num_gpus < 2) { printf("Need at least 2 GPUs.\n"); return 1; }

    int dev_count;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (num_gpus > dev_count) {
        printf("Requested %d GPUs but only %d available. Using %d.\n", num_gpus, dev_count, dev_count);
        num_gpus = dev_count;
    }

    enable_all_p2p(num_gpus);

    N = (N / num_gpus) * num_gpus;
    if (N <= 0) N = num_gpus;
    const int bytes = N * sizeof(float);
    printf("Running on %d GPUs, data size: %d floats (%.2f MB)\n", num_gpus, N, bytes / (1024.0 * 1024.0));

    // 分配数据
    float** h_data = (float**)malloc(num_gpus * sizeof(float*));
    float** d_data = (float**)malloc(num_gpus * sizeof(float*));
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc(&d_data[i], bytes));
        h_data[i] = (float*)malloc(bytes);
        fill_random(h_data[i], N);
        CUDA_CHECK(cudaMemcpy(d_data[i], h_data[i], bytes, cudaMemcpyHostToDevice));
    }

    int chunk_size = N / num_gpus;

    // 分配 send_buf 和 recv_buf（每个 rank 两个缓冲区）
    float** send_bufs = (float**)malloc(num_gpus * sizeof(float*));
    float** recv_bufs = (float**)malloc(num_gpus * sizeof(float*));
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc(&send_bufs[i], chunk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&recv_bufs[i], chunk_size * sizeof(float)));
    }

    // ========== Reduce-Scatter (步骤展开) ==========
    for (int step = 0; step < num_gpus - 1; ++step) {
        // 1. 所有 rank 将本轮的 send_chunk 拷贝到 send_buf，并推送到右邻居的 recv_buf
        for (int r = 0; r < num_gpus; ++r) {
            int send_chunk = (r - step + num_gpus) % num_gpus;
            int right = (r + 1) % num_gpus;

            CUDA_CHECK(cudaSetDevice(r));
            CUDA_CHECK(cudaMemcpy(send_bufs[r],
                                  d_data[r] + send_chunk * chunk_size,
                                  chunk_size * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpyPeer(recv_bufs[right], right,
                                      send_bufs[r], r,
                                      chunk_size * sizeof(float)));
        }

        // 同步
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // 2. 所有 rank 将 recv_buf 累加到本地 recv_chunk
        for (int r = 0; r < num_gpus; ++r) {
            int recv_chunk = (r - step - 1 + num_gpus) % num_gpus;

            CUDA_CHECK(cudaSetDevice(r));
            int threads = 256;
            int blocks = (chunk_size + threads - 1) / threads;
            simple_add_kernel<<<blocks, threads>>>(
                d_data[r] + recv_chunk * chunk_size,
                recv_bufs[r],
                chunk_size);
        }

        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // 此时每个 rank r 的 chunk r 是完全归约的结果

    // ========== All-Gather (步骤展开) ==========
    for (int step = 0; step < num_gpus - 1; ++step) {
        // 1. 所有 rank 将要发送的 chunk（已归约的）拷贝到 send_buf，推送到右邻居的 recv_buf
        for (int r = 0; r < num_gpus; ++r) {
            int send_chunk = (r - step + 1 + num_gpus) % num_gpus; // 注意这里是 +1
            int right = (r + 1) % num_gpus;

            CUDA_CHECK(cudaSetDevice(r));
            CUDA_CHECK(cudaMemcpy(send_bufs[r],
                                  d_data[r] + send_chunk * chunk_size,
                                  chunk_size * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpyPeer(recv_bufs[right], right,
                                      send_bufs[r], r,
                                      chunk_size * sizeof(float)));
        }

        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // 2. 所有 rank 将 recv_buf 覆盖本地 recv_chunk
        for (int r = 0; r < num_gpus; ++r) {
            int recv_chunk = (r - step + num_gpus) % num_gpus; // 注意这里没有 -1

            CUDA_CHECK(cudaSetDevice(r));
            CUDA_CHECK(cudaMemcpy(d_data[r] + recv_chunk * chunk_size,
                                  recv_bufs[r],
                                  chunk_size * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
        }

        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // 验证
    float* expected = (float*)malloc(bytes);
    for (int j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < num_gpus; ++i) sum += h_data[i][j];
        expected[j] = sum;
    }

    float max_err = 0.0f;
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        float* h_res = (float*)malloc(bytes);
        CUDA_CHECK(cudaMemcpy(h_res, d_data[i], bytes, cudaMemcpyDeviceToHost));
        for (int j = 0; j < N; ++j) {
            float err = fabsf(h_res[j] - expected[j]);
            if (err > max_err) max_err = err;
        }
        free(h_res);
    }
    printf("Max error across %d GPUs: %e\n", num_gpus, max_err);
    if (max_err < 1e-4) printf("✅ Ring AllReduce PASSED!\n");
    else printf("❌ FAILED!\n");

    // 清理
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaFree(d_data[i]));
        CUDA_CHECK(cudaFree(send_bufs[i]));
        CUDA_CHECK(cudaFree(recv_bufs[i]));
        free(h_data[i]);
    }
    free(h_data); free(d_data);
    free(send_bufs); free(recv_bufs);
    free(expected);
    return 0;
}
