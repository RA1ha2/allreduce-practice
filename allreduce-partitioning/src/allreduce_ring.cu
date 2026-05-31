#include "allreduce_ring.h"
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do {                               \
    cudaError_t err = call;                                 \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA error %s:%d: %s\n",          \
                __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while(0)

__global__ void simple_add_kernel(float* data, const float* recv, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += recv[idx];
}

void enable_all_p2p(int num_gpus) {
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        for (int j = 0; j < num_gpus; ++j) {
            if (i == j) continue;
            int canPeer = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&canPeer, i, j));
            if (!canPeer) {
                fprintf(stderr, "P2P %d -> %d not supported!\n", i, j);
                exit(EXIT_FAILURE);
            }
            cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
            if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                fprintf(stderr, "Failed to enable P2P %d->%d: %s\n", i, j, cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
        }
    }
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    printf("P2P enabled for all GPU pairs.\n");
}

// 全局发送/接收缓冲区，按需分配
static float** g_send_bufs = nullptr;
static float** g_recv_bufs = nullptr;
static int g_chunk_size = 0;

static void ensure_buffers(int num_gpus, int chunk_size) {
    if (g_send_bufs != nullptr && chunk_size <= g_chunk_size) return; // 足够大，无需重新分配

    // 释放旧缓冲区（如果存在）
    if (g_send_bufs != nullptr) {
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaFree(g_send_bufs[i]));
            CUDA_CHECK(cudaFree(g_recv_bufs[i]));
        }
        free(g_send_bufs);
        free(g_recv_bufs);
    }

    // 分配新的缓冲区
    g_send_bufs = (float**)malloc(num_gpus * sizeof(float*));
    g_recv_bufs = (float**)malloc(num_gpus * sizeof(float*));
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc(&g_send_bufs[i], chunk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_recv_bufs[i], chunk_size * sizeof(float)));
    }
    g_chunk_size = chunk_size;
}

void ring_reduce_scatter(float* data, int N, int rank, int num_gpus, float** all_data) {
    int chunk_size = N / num_gpus;
    if (chunk_size * num_gpus != N) {
        fprintf(stderr, "N must be divisible by num_gpus\n");
        exit(1);
    }

    ensure_buffers(num_gpus, chunk_size);
    float** send_bufs = g_send_bufs;
    float** recv_bufs = g_recv_bufs;

    for (int step = 0; step < num_gpus - 1; ++step) {
        for (int r = 0; r < num_gpus; ++r) {
            int send_chunk = (r - step + num_gpus) % num_gpus;
            int right = (r + 1) % num_gpus;
            CUDA_CHECK(cudaSetDevice(r));
            CUDA_CHECK(cudaMemcpy(send_bufs[r],
                                  all_data[r] + send_chunk * chunk_size,
                                  chunk_size * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpyPeer(recv_bufs[right], right,
                                      send_bufs[r], r,
                                      chunk_size * sizeof(float)));
        }
        for (int i = 0; i < num_gpus; ++i) { CUDA_CHECK(cudaSetDevice(i)); CUDA_CHECK(cudaDeviceSynchronize()); }

        for (int r = 0; r < num_gpus; ++r) {
            int recv_chunk = (r - step - 1 + num_gpus) % num_gpus;
            CUDA_CHECK(cudaSetDevice(r));
            int threads = 256, blocks = (chunk_size + threads - 1) / threads;
            simple_add_kernel<<<blocks, threads>>>(
                all_data[r] + recv_chunk * chunk_size,
                recv_bufs[r],
                chunk_size);
        }
        for (int i = 0; i < num_gpus; ++i) { CUDA_CHECK(cudaSetDevice(i)); CUDA_CHECK(cudaDeviceSynchronize()); }
    }
}

void ring_allgather(float* data, int N, int rank, int num_gpus, float** all_data) {
    int chunk_size = N / num_gpus;
    ensure_buffers(num_gpus, chunk_size);
    float** send_bufs = g_send_bufs;
    float** recv_bufs = g_recv_bufs;

    for (int step = 0; step < num_gpus - 1; ++step) {
        for (int r = 0; r < num_gpus; ++r) {
            int send_chunk = (r - step + 1 + num_gpus) % num_gpus;
            int right = (r + 1) % num_gpus;
            CUDA_CHECK(cudaSetDevice(r));
            CUDA_CHECK(cudaMemcpy(send_bufs[r],
                                  all_data[r] + send_chunk * chunk_size,
                                  chunk_size * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpyPeer(recv_bufs[right], right,
                                      send_bufs[r], r,
                                      chunk_size * sizeof(float)));
        }
        for (int i = 0; i < num_gpus; ++i) { CUDA_CHECK(cudaSetDevice(i)); CUDA_CHECK(cudaDeviceSynchronize()); }

        for (int r = 0; r < num_gpus; ++r) {
            int recv_chunk = (r - step + num_gpus) % num_gpus;
            CUDA_CHECK(cudaSetDevice(r));
            CUDA_CHECK(cudaMemcpy(all_data[r] + recv_chunk * chunk_size,
                                  recv_bufs[r],
                                  chunk_size * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
        }
        for (int i = 0; i < num_gpus; ++i) { CUDA_CHECK(cudaSetDevice(i)); CUDA_CHECK(cudaDeviceSynchronize()); }
    }
}

void ring_allreduce(float* data, int N, int rank, int num_gpus, float** all_data) {
    ring_reduce_scatter(data, N, rank, num_gpus, all_data);
    ring_allgather(data, N, rank, num_gpus, all_data);
}
