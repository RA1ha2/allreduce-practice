#pragma once
#include <cuda_runtime.h>

void enable_all_p2p(int num_gpus);
void ring_reduce_scatter(float* data, int N, int rank, int num_gpus, float** all_data);
void ring_allgather(float* data, int N, int rank, int num_gpus, float** all_data);
void ring_allreduce(float* data, int N, int rank, int num_gpus, float** all_data);
