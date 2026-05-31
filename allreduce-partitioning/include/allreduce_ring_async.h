#pragma once
#include <cuda_runtime.h>

// 流专用缓冲区结构体（完整定义）
struct StreamBuffers {
    float** send_bufs;
    float** recv_bufs;
    int chunk_size;
    int num_gpus;
    bool allocated;
};

void init_stream_buffers(StreamBuffers* bufs, int num_gpus, int chunk_size);
void free_stream_buffers(StreamBuffers* bufs);
void ring_reduce_scatter_async(float* data, int N, int rank, int num_gpus, float** all_data,
                               cudaStream_t stream, StreamBuffers* bufs);
void ring_allgather_async(float* data, int N, int rank, int num_gpus, float** all_data,
                          cudaStream_t stream, StreamBuffers* bufs);
