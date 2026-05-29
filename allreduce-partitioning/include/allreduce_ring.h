#pragma once
#include <cuda_runtime.h>

// 通用多 GPU 环形 AllReduce（支持任意 GPU 数量）
// data     : 当前 GPU 上的数据数组
// N        : 元素总数（必须能被 num_gpus 整除）
// rank     : 当前 GPU 编号 (0 ~ num_gpus-1)
// num_gpus : 总 GPU 数量
// all_data : 长度为 num_gpus 的设备指针数组，all_data[i] 为 rank i 的 data 指针
void ring_allreduce(float* data, int N, int rank, int num_gpus, float** all_data);
