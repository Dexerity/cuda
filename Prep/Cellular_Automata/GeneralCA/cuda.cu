#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "cuda_img.h"


// Kernel to clear the image
__global__ void kernel_run_sim_step(CudaPic inPic, CudaPic outPic, int ruleString1, int ruleString2)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is outside the image boundaries
    if(x >= inPic.m_size.x) return;
    if(y >= inPic.m_size.y) return;

    int bRule[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int sRule[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for(int i = 0; i < 8; i++)
    {
        
        bRule[i] = (int)(ruleString1 / powf(10, i)) % 10;
        sRule[i] = (int)(ruleString2 / powf(10, i)) % 10;
    }

    uchar1 tmp = inPic.getData<uchar1>(x, y);

    int alive = 0;
    int dead = 0;

    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            if (i == 0 && j == 0)
                continue;

            uchar1 tmp2;

            if(x + i < 0 || x + i >= inPic.m_size.x)
                tmp2 = make_uchar1(0);
            else if(y + j < 0 || y + j >= inPic.m_size.y)
                tmp2 = make_uchar1(0);
            else
                tmp2 = inPic.getData<uchar1>(x + i, y + j);

            if (tmp2.x == 255)
            {
                alive++;
            }
            else
            {
                dead++;
            }
        }
    }

    if(tmp.x == 255)
    {
        for(int i = 0; i < 8; i++)
        {
            if(sRule[i] == 0)
                continue;

            if(alive == sRule[i])
            {
                outPic.setData<uchar1>(x, y, make_uchar1(255));
                break;
            }
                outPic.setData<uchar1>(x, y, make_uchar1(0));
        }

        if(ruleString2 == 0)
            outPic.setData<uchar1>(x, y, make_uchar1(0));
    }
    else if(tmp.x == 0)
    {
        for(int i = 0; i < 8; i++)
        {
            if(bRule[i] == 0)
                continue;

            if(alive == bRule[i])
            {
                outPic.setData<uchar1>(x, y, make_uchar1(255));
                break;
            }
                outPic.setData<uchar1>(x, y, make_uchar1(0));
        }

        if(ruleString1 == 0)
            outPic.setData<uchar1>(x, y, make_uchar1(0));
    }
    else
    {
        outPic.setData<uchar1>(x, y, make_uchar1(0));
    }

}

__global__ void kernel_random_line(CudaPic inPic)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is outside the image boundaries
    if(x >= inPic.m_size.x) return;
    if(y >= inPic.m_size.y) return;

    if(y != 1) return;

    curandState state;
    curand_init(0, x, 0, &state);
    

    uchar l_new_pixel = curand(&state) % 2 * 255;
    inPic.setData<uchar1>(x, y, make_uchar1(l_new_pixel));
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

    if(random < 0.5)
    {
        pic.setData<uchar1>(x, y, make_uchar1(255));
    }
    else
    {
        pic.setData<uchar1>(x, y, make_uchar1(0));
    }
}

void cuda_run_sim_step(CudaPic inPic, CudaPic outPic, int ruleString, int ruleString2)
{
    cudaError_t l_cuda_err;

    // Set the block size
    int l_block_size = 32;

    // Calculate the number of blocks and threads
    dim3 l_blocks((inPic.m_size.x + l_block_size - 1) / l_block_size,
                  (inPic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_run_sim_step<<<l_blocks, l_threads>>>(inPic, outPic, ruleString, ruleString2);

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

void cuda_random_line(CudaPic inPic)
{
    cudaError_t l_cuda_err;

    // Set the block size
    int l_block_size = 32;

    // Calculate the number of blocks and threads
    dim3 l_blocks((inPic.m_size.x + l_block_size - 1) / l_block_size,
                  (inPic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_random_line<<<l_blocks, l_threads>>>(inPic);

    if((l_cuda_err = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cuda_err));
    }

    // Synchronize the device
    cudaDeviceSynchronize();
}

void cuda_create_stable(CudaPic pic, int x, int y)
{
    pic.setData<uchar1>(x, y, make_uchar1(255));
    pic.setData<uchar1>(x + 1, y, make_uchar1(255));
    pic.setData<uchar1>(x, y + 1, make_uchar1(255));
    pic.setData<uchar1>(x + 1, y + 1, make_uchar1(255));
}

void cuda_create_glider(CudaPic pic, int x, int y)
{
    pic.setData<uchar1>(x, y, make_uchar1(255));
    pic.setData<uchar1>(x + 1, y + 1, make_uchar1(255));
    pic.setData<uchar1>(x + 2, y + 1, make_uchar1(255));
    pic.setData<uchar1>(x + 2, y, make_uchar1(255));
    pic.setData<uchar1>(x + 2, y - 1, make_uchar1(255));
}
