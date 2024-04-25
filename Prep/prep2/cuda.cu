#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"


// Kernel to clear the image
__global__ void kernel_clear(CudaPic l_cv_in_pic)
{
    // Calculate the x and y coordinates of the thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is outside the image boundaries
    if(x >= l_cv_in_pic.m_size.x) return;
    if(y >= l_cv_in_pic.m_size.y) return;

    // Set the pixel value to 0
    l_cv_in_pic.setData<uchar1>(x, y, {0});
}

// Kernel to generate a black and white gradient circle
__global__ void kernel_BW_gradientCircle(CudaPic l_cv_in_pic)
{
    // Calculate the x and y coordinates of the thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is outside the image boundaries
    if(x >= l_cv_in_pic.m_size.x) return;
    if(y >= l_cv_in_pic.m_size.y) return;

    // Calculate the distance from the center of the image
    int cx = l_cv_in_pic.m_size.x / 2;
    int cy = l_cv_in_pic.m_size.y / 2;
    int distance = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));

    // Calculate the pixel value based on the distance
    unsigned char l_val = __sinf(distance / 20.0f) * 127 + 128;
    l_cv_in_pic.setData<uchar1>(x, y, {l_val});
}

// Kernel to generate a black and white horizontal gradient
__global__ void kernel_BW_gradientHor(CudaPic l_cv_in_pic)
{
    // Calculate the x and y coordinates of the thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is outside the image boundaries
    if(x >= l_cv_in_pic.m_size.x) return;
    if(y >= l_cv_in_pic.m_size.y) return;

    // Calculate the pixel value based on the x-coordinate
    unsigned char l_val = __sinf(x / 20.0f) * 127 + 128;
    l_cv_in_pic.setData<uchar1>(x, y, {l_val});
}

// Kernel to generate a black and white vertical gradient
__global__ void kernel_BW_gradientVer(CudaPic l_cv_in_pic)
{
    // Calculate the x and y coordinates of the thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is outside the image boundaries
    if(x >= l_cv_in_pic.m_size.x) return;
    if(y >= l_cv_in_pic.m_size.y) return;

    // Calculate the pixel value based on the y-coordinate
    unsigned char l_val = __sinf(y / 20.0f) * 127 + 128;
    l_cv_in_pic.setData<uchar1>(x, y, {l_val});
}

// Kernel to generate a checkerboard pattern
__global__ void kernel_checkerboard(CudaPic l_cv_in_pic)
{
    // Calculate the x and y coordinates of the thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is outside the image boundaries
    if(x >= l_cv_in_pic.m_size.x) return;
    if(y >= l_cv_in_pic.m_size.y) return;

    // Calculate the pixel value based on the block index
    unsigned char l_bw = 255 * ((blockIdx.x + blockIdx.y) & 1);

    l_cv_in_pic.setData<uchar1>(x, y, {l_bw});
}

// Function to clear the image using CUDA
void cuda_clear(CudaPic l_cv_in_pic)
{
    cudaError_t l_cuda_err;

    // Set the block size
    int l_block_size = 32;

    // Calculate the number of blocks and threads
    dim3 l_blocks((l_cv_in_pic.m_size.x + l_block_size - 1) / l_block_size,
                  (l_cv_in_pic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    // Launch the kernel to clear the image
    kernel_clear<<<l_blocks, l_threads>>>(l_cv_in_pic);

    // Check for any CUDA errors
    if((l_cuda_err = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cuda_err));
    }

    // Synchronize the device
    cudaDeviceSynchronize();
}

// Function to generate a black and white gradient circle using CUDA
void cuda_BW_gradientCircle(CudaPic l_cv_in_pic)
{
    cudaError_t l_cuda_err;

    // Set the block size
    int l_block_size = 32;

    // Calculate the number of blocks and threads
    dim3 l_blocks((l_cv_in_pic.m_size.x + l_block_size - 1) / l_block_size,
                  (l_cv_in_pic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    // Launch the kernel to generate the gradient circle
    kernel_BW_gradientCircle<<<l_blocks, l_threads>>>(l_cv_in_pic);

    // Check for any CUDA errors
    if((l_cuda_err = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cuda_err));
    }

    // Synchronize the device
    cudaDeviceSynchronize();
}

// Function to generate a black and white horizontal gradient using CUDA
void cuda_BW_gradientHor(CudaPic l_cv_in_pic)
{
    cudaError_t l_cuda_err;

    // Set the block size
    int l_block_size = 32;

    // Calculate the number of blocks and threads
    dim3 l_blocks((l_cv_in_pic.m_size.x + l_block_size - 1) / l_block_size,
                  (l_cv_in_pic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    // Launch the kernel to generate the horizontal gradient
    kernel_BW_gradientHor<<<l_blocks, l_threads>>>(l_cv_in_pic);

    // Check for any CUDA errors
    if((l_cuda_err = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cuda_err));
    }

    // Synchronize the device
    cudaDeviceSynchronize();
}

// Function to generate a black and white vertical gradient using CUDA
void cuda_BW_gradientVer(CudaPic l_cv_in_pic)
{
    cudaError_t l_cuda_err;

    // Set the block size
    int l_block_size = 32;

    // Calculate the number of blocks and threads
    dim3 l_blocks((l_cv_in_pic.m_size.x + l_block_size - 1) / l_block_size,
                  (l_cv_in_pic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    // Launch the kernel to generate the vertical gradient
    kernel_BW_gradientVer<<<l_blocks, l_threads>>>(l_cv_in_pic);

    // Check for any CUDA errors
    if((l_cuda_err = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cuda_err));
    }

    // Synchronize the device
    cudaDeviceSynchronize();
}

// Function to generate a checkerboard pattern using CUDA
void cuda_checkerboard(CudaPic l_cv_in_pic)
{
    cudaError_t l_cuda_err;

    // Set the block size
    int l_block_size = 32;

    // Calculate the number of blocks and threads
    dim3 l_blocks((l_cv_in_pic.m_size.x + l_block_size - 1) / l_block_size,
                  (l_cv_in_pic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    // Launch the kernel to generate the checkerboard pattern
    kernel_checkerboard<<<l_blocks, l_threads>>>(l_cv_in_pic);

    // Check for any CUDA errors
    if((l_cuda_err = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cuda_err));
    }

    // Synchronize the device
    cudaDeviceSynchronize();
}
