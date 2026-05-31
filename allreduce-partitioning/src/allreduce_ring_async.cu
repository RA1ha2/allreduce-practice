#include "allreduce_ring_async.h"
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

__global__ void simple_add_kernel_async(float* data, const float* recv, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += recv[idx];
}

void init_stream_buffers(StreamBuffers* bufs, int num_gpus, int chunk_size) {
    bufs->num_gpus = num_gpus;
    bufs->chunk_size = chunk_size;
    bufs->send_bufs = (float**)malloc(num_gpus * sizeof(float*));
    bufs->recv_bufs = (float**)malloc(num_gpus * sizeof(float*));
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc(&bufs->send_bufs[i], chunk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&bufs->recv_bufs[i], chunk_size * sizeof(float)));
    }
    bufs->allocated = true;
}

void free_stream_buffers(StreamBuffers* bufs) {
    if (!bufs->allocated) return;
    for (int i = 0; i < bufs->num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaFree(bufs->send_bufs[i]));
        CUDA_CHECK(cudaFree(bufs->recv_bufs[i]));
    }
    free(bufs->send_bufs);
    free(bufs->recv_bufs);
    bufs->allocated = false;
}

void ring_reduce_scatter_async(float* data, int N, int rank, int num_gpus, float** all_data,
                               cudaStream_t stream, StreamBuffers* bufs) {
    int chunk_size = N / num_gpus;
    if (chunk_size * num_gpus != N) {
        fprintf(stderr, "N must be divisible by num_gpus\n");
        exit(1);
    }

    if (!bufs->allocated || bufs->chunk_size < chunk_size) {
        if (bufs->allocated) free_stream_buffers(bufs);
        init_stream_buffers(bufs, num_gpus, chunk_size);
    }

    float** send_bufs = bufs->send_bufs;
    float** recv_bufs = bufs->recv_bufs;

    for (int step = 0; step < num_gpus - 1; ++step) {
        for (int r = 0; r < num_gpus; ++r) {
            int send_chunk = (r - step + num_gpus) % num_gpus;
            int right = (r + 1) % num_gpus;
            CUDA_CHECK(cudaSetDevice(r));
            CUDA_CHECK(cudaMemcpyAsync(send_bufs[r],
                                       all_data[r] + send_chunk * chunk_size,
                                       chunk_size * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyPeerAsync(recv_bufs[right], right,
                                           send_bufs[r], r,
                                           chunk_size * sizeof(float), stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (int r = 0; r < num_gpus; ++r) {
            int recv_chunk = (r - step - 1 + num_gpus) % num_gpus;
            CUDA_CHECK(cudaSetDevice(r));
            int threads = 256, blocks = (chunk_size + threads - 1) / threads;
            simple_add_kernel_async<<<blocks, threads, 0, stream>>>(
                all_data[r] + recv_chunk * chunk_size,
                recv_bufs[r],
                chunk_size);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

void ring_allgather_async(float* data, int N, int rank, int num_gpus, float** all_data,
                          cudaStream_t stream, StreamBuffers* bufs) {
    int chunk_size = N / num_gpus;
    if (!bufs->allocated || bufs->chunk_size < chunk_size) {
        if (bufs->allocated) free_stream_buffers(bufs);
        init_stream_buffers(bufs, num_gpus, chunk_size);
    }

    float** send_bufs = bufs->send_bufs;
    float** recv_bufs = bufs->recv_bufs;

    for (int step = 0; step < num_gpus - 1; ++step) {
        for (int r = 0; r < num_gpus; ++r) {
            int send_chunk = (r - step + 1 + num_gpus) % num_gpus;
            int right = (r + 1) % num_gpus;
            CUDA_CHECK(cudaSetDevice(r));
            CUDA_CHECK(cudaMemcpyAsync(send_bufs[r],
                                       all_data[r] + send_chunk * chunk_size,
                                       chunk_size * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyPeerAsync(recv_bufs[right], right,
                                           send_bufs[r], r,
                                           chunk_size * sizeof(float), stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (int r = 0; r < num_gpus; ++r) {
            int recv_chunk = (r - step + num_gpus) % num_gpus;
            CUDA_CHECK(cudaSetDevice(r));
            CUDA_CHECK(cudaMemcpyAsync(all_data[r] + recv_chunk * chunk_size,
                                       recv_bufs[r],
                                       chunk_size * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}
