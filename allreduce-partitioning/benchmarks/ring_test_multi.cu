#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "allreduce_ring.h"

#define CUDA_CHECK(call) do {                               \
    cudaError_t err = call;                                 \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA error %s:%d: %s\n",          \
                __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while(0)

// 用随机数填充数组
void fill_random(float* data, int N) {
    for (int i = 0; i < N; ++i)
        data[i] = static_cast<float>(rand()) / RAND_MAX;
}

int main(int argc, char** argv) {
    // 默认测试 4 张 GPU，可通过命令行参数指定
    int num_gpus = 4;
    if (argc > 1) num_gpus = atoi(argv[1]);
    if (num_gpus < 2) {
        printf("Need at least 2 GPUs.\n");
        return 1;
    }

    // 检查实际可用 GPU 数量
    int dev_count;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (num_gpus > dev_count) {
        printf("Requested %d GPUs but only %d available. Using %d.\n",
               num_gpus, dev_count, dev_count);
        num_gpus = dev_count;
    }
    printf("Running on %d GPUs...\n", num_gpus);

    // 1. 启用所有 GPU 间的 P2P 访问
 
    // 2. 数据大小：1M floats，保证能被 num_gpus 整除
    int base_N = 1024 * 1024;
    int N = (base_N / num_gpus) * num_gpus;  // 向下对齐
    const int bytes = N * sizeof(float);
    printf("Data size per GPU: %d floats (%.2f MB)\n", N, bytes / (1024.0 * 1024.0));

    // 3. 为每张 GPU 分配显存并初始化主机数据
    float** h_data = (float**)malloc(num_gpus * sizeof(float*));     // 主机端各 GPU 的原始数据
    float** d_data = (float**)malloc(num_gpus * sizeof(float*));     // 设备指针数组（主机侧保存）

    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc(&d_data[i], bytes));
        h_data[i] = (float*)malloc(bytes);
        fill_random(h_data[i], N);
        CUDA_CHECK(cudaMemcpy(d_data[i], h_data[i], bytes, cudaMemcpyHostToDevice));
    }

    // 4. 执行 Ring AllReduce（每张卡调用一次，主机端串行设置）
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        ring_allreduce(d_data[i], N, i, num_gpus, d_data);
    }

    // 等待所有 GPU 操作完成
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 5. 验证：所有 GPU 结果应相同，且等于所有原始数据之和
    // 计算 CPU 预期值
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
    if (max_err < 1e-4)
        printf("✅ Ring AllReduce PASSED!\n");
    else
        printf("❌ FAILED!\n");

    // 6. 清理资源
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaFree(d_data[i]));
        free(h_data[i]);
    }
    free(h_data);
    free(d_data);
    free(expected);
    return 0;
}
