#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"


// Kernel to clear the image
__global__ void kernel_flip(CudaPic inPic, CudaPic outPic, int dir)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is outside the image boundaries
    if(x >= inPic.m_size.x) return;
    if(y >= inPic.m_size.y) return;

    uchar3 tmp = inPic.getData<uchar3>(x, y);

    if(dir == 1)
    {
        int halfX = inPic.m_size.x / 2;
        if(x <= halfX)
            outPic.setData<uchar3>(x + halfX, y, tmp);
        else
            outPic.setData<uchar3>(x - halfX, y, tmp);
    }
    else if(dir == 2)
    {
        int halfY = inPic.m_size.y / 2;
        if(y <= halfY)
            outPic.setData<uchar3>(x, y + halfY, tmp);
        else
            outPic.setData<uchar3>(x, y - halfY, tmp);
    }
}

__global__ void kernel_color_remove(CudaPic inPic, CudaPic outPic, uchar3 color, double amount)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is outside the image boundaries
    if(x >= inPic.m_size.x) return;
    if(y >= inPic.m_size.y) return;

    uchar3 tmp = inPic.getData<uchar3>(x, y);
    uchar3 tmp2 = tmp;



    tmp2.x = (uchar)(tmp.x - color.x * (1 - amount));
    tmp2.y = (uchar)(tmp.y - color.y * (1 - amount));
    tmp2.z = (uchar)(tmp.z - color.z * (1 - amount));

    printf("Tmp: %d %d %d  =  tmp2: %d %d %d (amount: %f) (color: %d %d %d)\n", tmp.x, tmp.y, tmp.z, tmp2.x, tmp2.y, tmp2.z, amount, color.x, color.y, color.z);
    

    outPic.setData<uchar3>(x, y, tmp2);
}

void cuda_flip(CudaPic inPic, CudaPic outPic, int dir)
{
    cudaError_t l_cuda_err;

    // Set the block size
    int l_block_size = 32;

    // Calculate the number of blocks and threads
    dim3 l_blocks((inPic.m_size.x + l_block_size - 1) / l_block_size,
                  (inPic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    // Launch the kernel to clear the image
    kernel_flip<<<l_blocks, l_threads>>>(inPic, outPic, dir);

    // Check for any CUDA errors
    if((l_cuda_err = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cuda_err));
    }

    // Synchronize the device
    cudaDeviceSynchronize();
}

void cuda_color_remove(CudaPic inPic, CudaPic outPic, uchar3 color, double amount)
{
    cudaError_t l_cuda_err;

    // Set the block size
    int l_block_size = 32;

    // Calculate the number of blocks and threads
    dim3 l_blocks((inPic.m_size.x + l_block_size - 1) / l_block_size,
                  (inPic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    // Launch the kernel to clear the image
    kernel_color_remove<<<l_blocks, l_threads>>>(inPic, outPic, color, amount);

    // Check for any CUDA errors
    if((l_cuda_err = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cuda_err));
    }

    // Synchronize the device
    cudaDeviceSynchronize();
}