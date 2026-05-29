#pragma once
#include <cuda_runtime.h>

// 启动向量化 elementwise 加法
// a, b: 输入数组（设备指针）
// c   : 输出数组（可与 a 或 b 相同，实现原地操作）
// n   : 数组元素个数
// stream: CUDA 流（默认为空流）
void launch_elementwise_add(const float* a, const float* b, float* c, int n, cudaStream_t stream = 0);