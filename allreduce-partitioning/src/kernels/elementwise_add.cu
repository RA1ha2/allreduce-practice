#include "kernels/elementwise_add.h"

// 向量化逐元素加法内核：每个线程一次处理 4 个 float
__global__ void elementwise_add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < n) {
        float4 a4 = reinterpret_cast<const float4*>(a)[idx / 4];
        float4 b4 = reinterpret_cast<const float4*>(b)[idx / 4];
        float4 c4;
        c4.x = a4.x + b4.x;
        c4.y = a4.y + b4.y;
        c4.z = a4.z + b4.z;
        c4.w = a4.w + b4.w;
        reinterpret_cast<float4*>(c)[idx / 4] = c4;
    } else {
        // 处理尾部不足 4 的倍数部分
        for (int i = idx; i < n; ++i) {
            c[i] = a[i] + b[i];
        }
    }
}

void launch_elementwise_add(const float* a, const float* b, float* c, int n, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (n / 4 + block_size - 1) / block_size;
    elementwise_add_kernel<<<grid_size, block_size, 0, stream>>>(a, b, c, n);
}