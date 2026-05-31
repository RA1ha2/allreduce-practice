#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "allreduce_ring.h"

#define CUDA_CHECK(call) do {                               \
    cudaError_t err = call;                                 \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA error %s:%d: %s\n",          \
                __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while(0)

// 声明调试函数（将在 allreduce_ring_debug.cu 中实现）
void ring_reduce_scatter_debug(float* data, int N, int rank, int num_gpus, float** all_data);
void ring_allgather_debug(float* data, int N, int rank, int num_gpus, float** all_data);

int main() {
    int num_gpus = 4;
    int N = 16;          // 每张卡16个float，可被4整除，chunk_size=4
    const int bytes = N * sizeof(float);

    enable_all_p2p(num_gpus);

    // 分配并初始化已知数据：GPU k 全为 (k+1)
    float** h_data = (float**)malloc(num_gpus * sizeof(float*));
    float** d_data = (float**)malloc(num_gpus * sizeof(float*));
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc(&d_data[i], bytes));
        h_data[i] = (float*)malloc(bytes);
        float val = (float)(i + 1);
        for (int j = 0; j < N; ++j) h_data[i][j] = val;
        CUDA_CHECK(cudaMemcpy(d_data[i], h_data[i], bytes, cudaMemcpyHostToDevice));
    }

    printf("Initial data:\n");
    for (int i = 0; i < num_gpus; ++i) {
        printf("GPU %d: ", i);
        for (int j = 0; j < N; ++j) printf("%.1f ", h_data[i][j]);
        printf("\n");
    }

    // Reduce-Scatter with debug
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        ring_reduce_scatter_debug(d_data[i], N, i, num_gpus, d_data);
    }
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    printf("\nAfter Reduce-Scatter:\n");
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        float* tmp = (float*)malloc(bytes);
        CUDA_CHECK(cudaMemcpy(tmp, d_data[i], bytes, cudaMemcpyDeviceToHost));
        printf("GPU %d: ", i);
        for (int j = 0; j < N; ++j) printf("%.1f ", tmp[j]);
        printf("\n");
        free(tmp);
    }

    // All-Gather with debug
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        ring_allgather_debug(d_data[i], N, i, num_gpus, d_data);
    }
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    printf("\nFinal data:\n");
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        float* tmp = (float*)malloc(bytes);
        CUDA_CHECK(cudaMemcpy(tmp, d_data[i], bytes, cudaMemcpyDeviceToHost));
        printf("GPU %d: ", i);
        for (int j = 0; j < N; ++j) printf("%.1f ", tmp[j]);
        printf("\n");
        free(tmp);
    }

    float expected = 10.0f; // 1+2+3+4 = 10
    float max_err = 0.0f;
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        float* tmp = (float*)malloc(bytes);
        CUDA_CHECK(cudaMemcpy(tmp, d_data[i], bytes, cudaMemcpyDeviceToHost));
        for (int j = 0; j < N; ++j) {
            float err = fabsf(tmp[j] - expected);
            if (err > max_err) max_err = err;
        }
        free(tmp);
    }
    printf("Max error: %f\n", max_err);
    if (max_err < 1e-4) printf("✅ PASSED!\n");
    else printf("❌ FAILED!\n");

    for (int i = 0; i < num_gpus; ++i) { CUDA_CHECK(cudaSetDevice(i)); CUDA_CHECK(cudaFree(d_data[i])); free(h_data[i]); }
    free(h_data); free(d_data);
    return 0;
}
