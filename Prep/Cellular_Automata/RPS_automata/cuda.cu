#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "cuda_img.h"


// Kernel to clear the image
__global__ void kernel_run_sim_step(CudaPic inPic, CudaPic outPic)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is outside the image boundaries
    if(x >= inPic.m_size.x) return;
    if(y >= inPic.m_size.y) return;

    uchar3 tmp = inPic.getData<uchar3>(x, y);

    int rock = 0;
    int paper = 0;
    int scissors = 0;

    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            if (i == 0 && j == 0)
                continue;

            if (i + x < 0 || i + x >= inPic.m_size.x || j + y < 0 || j + y >= inPic.m_size.y)
            continue;

            uchar3 tmp2 = inPic.getData<uchar3>(x + i, y + j);
            if (tmp2.x == 255)
            {
                rock++;
            }
            else if (tmp2.y == 255)
            {
                paper++;
            }
            else if (tmp2.z == 255)
            {
                scissors++;
            }
        }
    }

    if(tmp.x == 255)
    {
        if(paper > 2)
            outPic.setData<uchar3>(x, y, make_uchar3(0, 255, 0));
        else
            outPic.setData<uchar3>(x, y, tmp);
    }

    if(tmp.y == 255)
    {
        if(scissors > 2)
            outPic.setData<uchar3>(x, y, make_uchar3(0, 0, 255));
        else
            outPic.setData<uchar3>(x, y, tmp);
    }

    if(tmp.z == 255)
    {
        if(rock > 2)
            outPic.setData<uchar3>(x, y, make_uchar3(255, 0, 0));
        else
            outPic.setData<uchar3>(x, y, make_uchar3(0, 0, 255));
    }

}

__global__ void kernel_random_canvas(CudaPic pic)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is outside the image boundaries
    if(x >= pic.m_size.x) return;
    if(y >= pic.m_size.y) return;

    // Generate a random number between 0 and 1
    curandState state;
    curand_init(0, x, 0, &state);
    float random = curand_uniform(&state);

    if(random < 0.33)
    {
        pic.setData<uchar3>(x, y, make_uchar3(255, 0, 0));
    }
    else if(random < 0.66)
    {
        pic.setData<uchar3>(x, y, make_uchar3(0, 255, 0));
    }
    else
    {
        pic.setData<uchar3>(x, y, make_uchar3(0, 0, 255));
    }
}

void cuda_run_sim_step(CudaPic inPic, CudaPic outPic)
{
    cudaError_t l_cuda_err;

    // Set the block size
    int l_block_size = 32;

    // Calculate the number of blocks and threads
    dim3 l_blocks((inPic.m_size.x + l_block_size - 1) / l_block_size,
                  (inPic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_run_sim_step<<<l_blocks, l_threads>>>(inPic, outPic);

    if((l_cuda_err = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cuda_err));
    }

    // Synchronize the device
    cudaDeviceSynchronize();
}

void cuda_random_canvas(CudaPic pic)
{
    cudaError_t l_cuda_err;

    // Set the block size
    int l_block_size = 32;

    // Calculate the number of blocks and threads
    dim3 l_blocks((pic.m_size.x + l_block_size - 1) / l_block_size,
                  (pic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_random_canvas<<<l_blocks, l_threads>>>(pic);

    if((l_cuda_err = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cuda_err));
    }

    // Synchronize the device
    cudaDeviceSynchronize();
}