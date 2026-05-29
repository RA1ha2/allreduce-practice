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

    // Lambda：获取指定 GPU 上第 chunk_idx 块数据的指针（所有 GPU 均已开启 P2P）
    auto peer_ptr = [&](int peer_rank, int chunk_idx) -> float* {
        return all_data[peer_rank] + chunk_idx * chunk_size;
    };

    // ==================== Reduce-Scatter 阶段 ====================
    for (int step = 0; step < num_gpus - 1; ++step) {
        // 当前步本 rank 要发送的 chunk 编号
        int send_chunk = (rank - step + num_gpus) % num_gpus;
        // 当前步本 rank 要接收并归约的 chunk 编号
        int recv_chunk = (rank - step - 1 + num_gpus) % num_gpus;
        int peer = (rank + 1) % num_gpus;   // 环中的右邻居

        // 从 peer 拷贝它的 send_chunk 到本地 recv_buf
        CUDA_CHECK(cudaMemcpyAsync(
            recv_buf,
            peer_ptr(peer, send_chunk),
            chunk_size * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream
        ));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // 将收到的数据累加到本地的 recv_chunk
        launch_elementwise_add(
            data + recv_chunk * chunk_size,
            recv_buf,
            data + recv_chunk * chunk_size,
            chunk_size,
            stream
        );
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // ==================== All-Gather 阶段 ====================
    for (int step = 0; step < num_gpus - 1; ++step) {
        int send_chunk = (rank - step + 1 + num_gpus) % num_gpus;
        int recv_chunk = (rank - step + num_gpus) % num_gpus;
        int peer = (rank + 1) % num_gpus;

        // 从 peer 拷贝对方已完全归约的 send_chunk
        CUDA_CHECK(cudaMemcpyAsync(
            recv_buf,
            peer_ptr(peer, send_chunk),
            chunk_size * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream
        ));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // 覆盖本地 recv_chunk
        CUDA_CHECK(cudaMemcpyAsync(
            data + recv_chunk * chunk_size,
            recv_buf,
            chunk_size * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream
        ));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    CUDA_CHECK(cudaFree(recv_buf));
    CUDA_CHECK(cudaStreamDestroy(stream));
}