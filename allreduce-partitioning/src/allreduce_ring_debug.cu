#include "allreduce_ring.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

static void p2p_pull_debug(float* dst, int dst_dev, const float* src, int src_dev, size_t bytes, const char* label) {
    CUDA_CHECK(cudaSetDevice(dst_dev));
    CUDA_CHECK(cudaMemcpyPeer(dst, dst_dev, src, src_dev, bytes));
    if (bytes >= 4 * sizeof(float)) {
        float* host_tmp = (float*)malloc(bytes);
        CUDA_CHECK(cudaMemcpy(host_tmp, dst, bytes, cudaMemcpyDeviceToHost));
        printf("%s received (first 4): %f %f %f %f\n", label, host_tmp[0], host_tmp[1], host_tmp[2], host_tmp[3]);
        free(host_tmp);
    }
}

void ring_reduce_scatter_debug(float* data, int N, int rank, int num_gpus, float** all_data) {
    int chunk_size = N / num_gpus;
    CUDA_CHECK(cudaSetDevice(rank));
    float* recv_buf;
    CUDA_CHECK(cudaMalloc(&recv_buf, chunk_size * sizeof(float)));

    auto left_peer = [&](int r) { return (r - 1 + num_gpus) % num_gpus; };
    auto chunk_ptr = [&](int dev, int chunk) -> float* {
        return all_data[dev] + chunk * chunk_size;
    };

    printf("\nRank %d Reduce-Scatter (chunk_size=%d)\n", rank, chunk_size);
    for (int step = 0; step < num_gpus - 1; ++step) {
        int recv_chunk = (rank - step - 1 + num_gpus) % num_gpus;
        int peer = left_peer(rank);
        printf("Rank %d step %d: pulling chunk %d from peer %d\n", rank, step, recv_chunk, peer);
        p2p_pull_debug(recv_buf, rank, chunk_ptr(peer, recv_chunk), peer, chunk_size * sizeof(float), "p2p_pull");

        // 打印本地 chunk 归约前
        float* host_chunk = (float*)malloc(chunk_size * sizeof(float));
        CUDA_CHECK(cudaMemcpy(host_chunk, data + recv_chunk * chunk_size, chunk_size * sizeof(float), cudaMemcpyDeviceToHost));
        printf("  local chunk %d before add (first 4): %f %f %f %f\n", recv_chunk, host_chunk[0], host_chunk[1], host_chunk[2], host_chunk[3]);

        int threads = 256, blocks = (chunk_size + threads - 1) / threads;
        simple_add_kernel<<<blocks, threads>>>(data + recv_chunk * chunk_size, recv_buf, chunk_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(host_chunk, data + recv_chunk * chunk_size, chunk_size * sizeof(float), cudaMemcpyDeviceToHost));
        printf("  local chunk %d after add  (first 4): %f %f %f %f\n", recv_chunk, host_chunk[0], host_chunk[1], host_chunk[2], host_chunk[3]);
        free(host_chunk);
    }
    CUDA_CHECK(cudaFree(recv_buf));
}

void ring_allgather_debug(float* data, int N, int rank, int num_gpus, float** all_data) {
    int chunk_size = N / num_gpus;
    CUDA_CHECK(cudaSetDevice(rank));
    float* recv_buf;
    CUDA_CHECK(cudaMalloc(&recv_buf, chunk_size * sizeof(float)));

    auto left_peer = [&](int r) { return (r - 1 + num_gpus) % num_gpus; };
    auto chunk_ptr = [&](int dev, int chunk) -> float* {
        return all_data[dev] + chunk * chunk_size;
    };

    printf("\nRank %d All-Gather\n", rank);
    for (int step = 0; step < num_gpus - 1; ++step) {
        int recv_chunk = (rank - step + num_gpus) % num_gpus;
        int peer = left_peer(rank);
        printf("Rank %d step %d: pulling chunk %d from peer %d\n", rank, step, recv_chunk, peer);
        p2p_pull_debug(recv_buf, rank, chunk_ptr(peer, recv_chunk), peer, chunk_size * sizeof(float), "p2p_pull");

        CUDA_CHECK(cudaMemcpy(data + recv_chunk * chunk_size, recv_buf, chunk_size * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        float* host_chunk = (float*)malloc(chunk_size * sizeof(float));
        CUDA_CHECK(cudaMemcpy(host_chunk, data + recv_chunk * chunk_size, chunk_size * sizeof(float), cudaMemcpyDeviceToHost));
        printf("  local chunk %d after copy (first 4): %f %f %f %f\n", recv_chunk, host_chunk[0], host_chunk[1], host_chunk[2], host_chunk[3]);
        free(host_chunk);
    }
    CUDA_CHECK(cudaFree(recv_buf));
}
