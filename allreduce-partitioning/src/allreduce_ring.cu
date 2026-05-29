#include "allreduce_ring.h"
#include "kernels/elementwise_add.h"
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

void ring_allreduce(float* data, int N, int rank, int num_gpus, float** all_data) {
    int chunk_size = N / num_gpus;
    float* recv_buf;
    CUDA_CHECK(cudaMalloc(&recv_buf, chunk_size * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 左邻居
    auto left_peer = [&](int r) { return (r - 1 + num_gpus) % num_gpus; };
    // 获取指定 GPU 上第 chunk_idx 块的指针
    auto chunk_ptr = [&](int dev, int chunk_idx) -> float* {
        return all_data[dev] + chunk_idx * chunk_size;
    };

    // ==================== Reduce-Scatter ====================
    for (int step = 0; step < num_gpus - 1; ++step) {
        int recv_chunk = (rank - step - 1 + num_gpus) % num_gpus; // 本 rank 要接收并归约的块
        int peer = left_peer(rank);

        // 从左邻居拉取它的 recv_chunk（即邻居本次要发送的块）
        CUDA_CHECK(cudaMemcpyPeerAsync(
            recv_buf, rank,
            chunk_ptr(peer, recv_chunk), peer,
            chunk_size * sizeof(float),
            stream
        ));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        launch_elementwise_add(
            data + recv_chunk * chunk_size,
            recv_buf,
            data + recv_chunk * chunk_size,
            chunk_size,
            stream
        );
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // ==================== All-Gather ====================
    for (int step = 0; step < num_gpus - 1; ++step) {
        int recv_chunk = (rank - step + num_gpus) % num_gpus; // 本 rank 要接收并覆盖的块
        int peer = left_peer(rank);

        // 从左邻居拉取它的 recv_chunk（邻居要发送的块，恰好等于本地的 recv_chunk）
        CUDA_CHECK(cudaMemcpyPeerAsync(
            recv_buf, rank,
            chunk_ptr(peer, recv_chunk), peer,
            chunk_size * sizeof(float),
            stream
        ));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // 覆盖本地 recv_chunk
        CUDA_CHECK(cudaMemcpyPeerAsync(
            data + recv_chunk * chunk_size, rank,
            recv_buf, rank,
            chunk_size * sizeof(float),
            stream
        ));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    CUDA_CHECK(cudaFree(recv_buf));
    CUDA_CHECK(cudaStreamDestroy(stream));
}