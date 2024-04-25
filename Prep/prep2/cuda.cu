#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"


__global__ void kernel_clear(CudaPic l_cv_in_pic)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= l_cv_in_pic.m_size.x) return;
    if(y >= l_cv_in_pic.m_size.y) return;

    l_cv_in_pic.setData<uchar1>(x, y, {0});
}

__global__ void kernel_BW_gradientCircle(CudaPic l_cv_in_pic)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= l_cv_in_pic.m_size.x) return;
    if(y >= l_cv_in_pic.m_size.y) return;

    int cx = l_cv_in_pic.m_size.x / 2;
    int cy = l_cv_in_pic.m_size.y / 2;

    int distance = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));

    unsigned char l_val = __sinf(distance / 20.0f) * 127 + 128;
    l_cv_in_pic.setData<uchar1>(x, y, {l_val});
}

__global__ void kernel_BW_gradientHor(CudaPic l_cv_in_pic)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= l_cv_in_pic.m_size.x) return;
    if(y >= l_cv_in_pic.m_size.y) return;

    unsigned char l_val = __sinf(x / 20.0f) * 127 + 128;
    l_cv_in_pic.setData<uchar1>(x, y, {l_val});
}

__global__ void kernel_BW_gradientVer(CudaPic l_cv_in_pic)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= l_cv_in_pic.m_size.x) return;
    if(y >= l_cv_in_pic.m_size.y) return;

    unsigned char l_val = __sinf(y / 20.0f) * 127 + 128;
    l_cv_in_pic.setData<uchar1>(x, y, {l_val});
}

__global__ void kernel_checkerboard(CudaPic l_cv_in_pic)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= l_cv_in_pic.m_size.x) return;
    if(y >= l_cv_in_pic.m_size.y) return;

    unsigned char l_bw = 255 * ((blockIdx.x + blockIdx.y) & 1);

    l_cv_in_pic.setData<uchar1>(x, y, {l_bw});
}

void cuda_clear(CudaPic l_cv_in_pic)
{
    cudaError_t l_cuda_err;

    int l_block_size = 32;

    dim3 l_blocks((l_cv_in_pic.m_size.x + l_block_size - 1) / l_block_size,
                  (l_cv_in_pic.m_size.y + l_block_size - 1) / l_block_size);

    dim3 l_threads(l_block_size, l_block_size);

    kernel_clear<<<l_blocks, l_threads>>>(l_cv_in_pic);

    if((l_cuda_err = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cuda_err));
    }

    cudaDeviceSynchronize();
}

void cuda_BW_gradientCircle(CudaPic l_cv_in_pic)
{
    cudaError_t l_cuda_err;

    int l_block_size = 32;

    dim3 l_blocks((l_cv_in_pic.m_size.x + l_block_size - 1) / l_block_size,
                  (l_cv_in_pic.m_size.y + l_block_size - 1) / l_block_size);

    dim3 l_threads(l_block_size, l_block_size);

    kernel_BW_gradientCircle<<<l_blocks, l_threads>>>(l_cv_in_pic);

    if((l_cuda_err = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cuda_err));
    }

    cudaDeviceSynchronize();
}

void cuda_BW_gradientHor(CudaPic l_cv_in_pic)
{
    cudaError_t l_cuda_err;

    int l_block_size = 32;

    dim3 l_blocks((l_cv_in_pic.m_size.x + l_block_size - 1) / l_block_size,
                  (l_cv_in_pic.m_size.y + l_block_size - 1) / l_block_size);

    dim3 l_threads(l_block_size, l_block_size);

    kernel_BW_gradientHor<<<l_blocks, l_threads>>>(l_cv_in_pic);

    if((l_cuda_err = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cuda_err));
    }

    cudaDeviceSynchronize();
}

void cuda_BW_gradientVer(CudaPic l_cv_in_pic)
{
    cudaError_t l_cuda_err;

    int l_block_size = 32;

    dim3 l_blocks((l_cv_in_pic.m_size.x + l_block_size - 1) / l_block_size,
                  (l_cv_in_pic.m_size.y + l_block_size - 1) / l_block_size);

    dim3 l_threads(l_block_size, l_block_size);

    kernel_BW_gradientVer<<<l_blocks, l_threads>>>(l_cv_in_pic);

    if((l_cuda_err = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cuda_err));
    }

    cudaDeviceSynchronize();
}

void cuda_checkerboard(CudaPic l_cv_in_pic)
{
    cudaError_t l_cuda_err;

    int l_block_size = 32;

    dim3 l_blocks((l_cv_in_pic.m_size.x + l_block_size - 1) / l_block_size,
                  (l_cv_in_pic.m_size.y + l_block_size - 1) / l_block_size);

    dim3 l_threads(l_block_size, l_block_size);

    kernel_checkerboard<<<l_blocks, l_threads>>>(l_cv_in_pic);

    if((l_cuda_err = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cuda_err));
    }

    cudaDeviceSynchronize();
}
