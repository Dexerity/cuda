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

__global__ void kernel_rainbowGradient(CudaPic l_cv_in_pic)
{
    // Calculate the x and y coordinates of the thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is outside the image boundaries
    if(x >= l_cv_in_pic.m_size.x) return;
    if(y >= l_cv_in_pic.m_size.y) return;

    uchar3 colors[7] = {
        {0, 0, 255},
        {0, 255, 255},
        {0, 255, 0},
        {255, 255, 0},
        {255, 0, 0},
        {255, 0, 255},
        {0, 0, 255}
    };

    int colorIndex = x / ((double)l_cv_in_pic.m_size.x / 6);
    double colorWeight = (double)x / ((double)l_cv_in_pic.m_size.x / 6.0) - colorIndex;
    double Yalpha = (double)y / (double)l_cv_in_pic.m_size.y;
    
    uchar3 color1 = colors[colorIndex];
    uchar3 color2 = colors[colorIndex + 1];

    uchar4 color = {
        (uchar)(color1.x * (1 - colorWeight) + color2.x * colorWeight),
        (uchar)(color1.y * (1 - colorWeight) + color2.y * colorWeight),
        (uchar)(color1.z * (1 - colorWeight) + color2.z * colorWeight),
        (uchar)((1 - Yalpha) * 255)
    };

    l_cv_in_pic.setData<uchar4>(x, y, color);

    if(y == 1)
    {
        printf("Color: %d, %d, %d, %d:  %d,%d   %f  (%d, %d)\n", color.x, color.y, color.z, color.w, x, y, Yalpha, l_cv_in_pic.m_size.x, l_cv_in_pic.m_size.y);
    }
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

void cuda_rainbowGradient(CudaPic l_cv_in_pic)
{
    cudaError_t l_cuda_err;

    // Set the block size
    int l_block_size = 32;

    // Calculate the number of blocks and threads
    dim3 l_blocks((l_cv_in_pic.m_size.x + l_block_size - 1) / l_block_size,
                  (l_cv_in_pic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    // Launch the kernel to clear the image
    kernel_rainbowGradient<<<l_blocks, l_threads>>>(l_cv_in_pic);

    // Check for any CUDA errors
    if((l_cuda_err = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cuda_err));
    }

    // Synchronize the device
    cudaDeviceSynchronize();
}

