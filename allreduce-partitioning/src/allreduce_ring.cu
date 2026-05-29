#include "allreduce_ring.h"
#include "kernels/elementwise_add.h"  // 提供 launch_elementwise_add 声明
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

// 环形 AllReduce（推送模式，向右发送，从左接收）
// 假设所有 GPU 已启用 P2P 访问，N 可被 num_gpus 整除
void ring_allreduce(float* data, int N, int rank, int num_gpus, float** all_data) {
    int chunk_size = N / num_gpus;
    float* recv_buf;
    CUDA_CHECK(cudaMalloc(&recv_buf, chunk_size * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int left  = (rank - 1 + num_gpus) % num_gpus;   // 左邻居
    int right = (rank + 1) % num_gpus;               // 右邻居

    // ==================== Reduce-Scatter 阶段 ====================
    for (int step = 0; step < num_gpus - 1; ++step) {
        // 本步要发送给右邻居的 chunk 编号
        int send_chunk = (rank - step + num_gpus) % num_gpus;
        // 本步要从左邻居接收并归约的 chunk 编号
        int recv_chunk = (rank - step - 1 + num_gpus) % num_gpus;

        // 1. 将本地 send_chunk 推送到右邻居的对应位置
        CUDA_CHECK(cudaMemcpyPeerAsync(
            all_data[right] + send_chunk * chunk_size,  // 右邻居的 send_chunk 地址
            right,                                      // 右邻居设备号
            data + send_chunk * chunk_size,             // 本地 send_chunk 数据
            rank,                                       // 本设备号
            chunk_size * sizeof(float),
            stream
        ));

        // 2. 从左邻居接收它的 send_chunk (即本地的 recv_chunk) 到 recv_buf
        CUDA_CHECK(cudaMemcpyPeerAsync(
            recv_buf,                                   // 本地接收缓冲区
            rank,
            all_data[left] + recv_chunk * chunk_size,   // 左邻居的 recv_chunk 数据
            left,                                       // 左邻居设备号
            chunk_size * sizeof(float),
            stream
        ));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // 3. 将收到的数据累加到本地 recv_chunk
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

        // 1. 推送本地已归约的 send_chunk 到右邻居
        CUDA_CHECK(cudaMemcpyPeerAsync(
            all_data[right] + send_chunk * chunk_size,
            right,
            data + send_chunk * chunk_size,
            rank,
            chunk_size * sizeof(float),
            stream
        ));

        // 2. 从左邻居接收其 send_chunk (即本地的 recv_chunk)
        CUDA_CHECK(cudaMemcpyPeerAsync(
            recv_buf,
            rank,
            all_data[left] + recv_chunk * chunk_size,
            left,
            chunk_size * sizeof(float),
            stream
        ));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // 3. 用接收到的数据覆盖本地 recv_chunk（无需累加）
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