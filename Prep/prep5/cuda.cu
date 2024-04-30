#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"


// Kernel to clear the image
__global__ void kernel_sep_rotate(CudaPic inPic, CudaPic outPic)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is outside the image boundaries
    if(x >= inPic.m_size.x) return;
    if(y >= inPic.m_size.y) return;

    int halfX = inPic.m_size.x / 2;
    int halfY = inPic.m_size.y / 2;

    if(x < halfX)
    {
        if(y < halfY)
            outPic.setData<uchar3>(y, halfY - x, inPic.getData<uchar3>(x, y));
        else
            outPic.setData<uchar3>(y - halfX, (halfY - x) + halfY, inPic.getData<uchar3>(x, y));
    }
    else
    {
        if(y < halfY)
            outPic.setData<uchar3>(y + halfX, halfY + (halfX - x), inPic.getData<uchar3>(x, y));
        else
            outPic.setData<uchar3>(y, inPic.m_size.y - x + halfY, inPic.getData<uchar3>(x, y));
    }
}

__global__ void kernel_remove_color_quadrant(CudaPic inPic, CudaPic outPic)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is outside the image boundaries
    if(x >= inPic.m_size.x) return;
    if(y >= inPic.m_size.y) return;

    int halfX = inPic.m_size.x / 2;
    int halfY = inPic.m_size.y / 2;

    uchar3 tmp = inPic.getData<uchar3>(x, y);
    

    if(x < halfX)
    {
        if(y < halfY)
            outPic.setData<uchar3>(x, y, make_uchar3(0, tmp.y, tmp.z));
        else
            outPic.setData<uchar3>(x, y, make_uchar3(tmp.x, 0, tmp.z));
    }
    else
    {
        if(y < halfY)
            outPic.setData<uchar3>(x, y, make_uchar3(tmp.x, tmp.y, 0));
        else
            outPic.setData<uchar3>(x, y, make_uchar3(tmp.x, tmp.y, tmp.z));
    }
    
}

void cuda_sep_rotate(CudaPic inPic, CudaPic outPic)
{
    cudaError_t l_cuda_err;

    // Set the block size
    int l_block_size = 32;

    // Calculate the number of blocks and threads
    dim3 l_blocks((inPic.m_size.x + l_block_size - 1) / l_block_size,
                  (inPic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_sep_rotate<<<l_blocks, l_threads>>>(inPic, outPic);

    if((l_cuda_err = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cuda_err));
    }

    // Synchronize the device
    cudaDeviceSynchronize();
}

void cuda_remove_color_quadrant(CudaPic inPic, CudaPic outPic)
{
    cudaError_t l_cuda_err;

    // Set the block size
    int l_block_size = 32;

    // Calculate the number of blocks and threads
    dim3 l_blocks((inPic.m_size.x + l_block_size - 1) / l_block_size,
                  (inPic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_remove_color_quadrant<<<l_blocks, l_threads>>>(inPic, outPic);

    if((l_cuda_err = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cuda_err));
    }

    // Synchronize the device
    cudaDeviceSynchronize();
}