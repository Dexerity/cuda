#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "cuda_img.h"

bool elementary_rule(uchar inRule, bool inLeft, bool inCenter, bool inRight);

__global__ void kernel_random_line(CudaPic inPic)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is outside the image boundaries
    if(x >= inPic.m_size.x) return;
    if(y >= inPic.m_size.y) return;

    if(y != 0) return;

    curandState state;
    curand_init(0, x, 0, &state);
    

    uchar l_new_pixel = curand(&state) % 2 * 255;
    inPic.setData<uchar1>(x, y, make_uchar1(l_new_pixel));
}

// Kernel to clear the image
__global__ void kernel_run_sim_step(CudaPic inPic, uchar inRule, int line)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is outside the image boundaries
    if(x >= inPic.m_size.x) return;
    if(y >= inPic.m_size.y) return;

    if(y != line) return;



    uchar1 l_pixel = inPic.getData<uchar1>(x, y - 1);

    if(x == 0) 
    {
        return;
    }
    else if(x == inPic.m_size.x - 1)
    {
        return;
    }
    else
    {
        uchar l_left = inPic.getData<uchar1>(x - 1, y - 1).x;
        uchar l_center = inPic.getData<uchar1>(x, y - 1).x;
        uchar l_right = inPic.getData<uchar1>(x + 1, y - 1).x;

        uchar l_new_pixel = elementary_rule(inRule, l_left, l_center, l_right) * 255;
        inPic.setData<uchar1>(x, y, make_uchar1(l_new_pixel));
    }

}

void cuda_run_sim_step(CudaPic inPic, uchar inRule, int line)
{
    cudaError_t l_cuda_err;

    // Set the block size
    int l_block_size = 32;

    // Calculate the number of blocks and threads
    dim3 l_blocks((inPic.m_size.x + l_block_size - 1) / l_block_size,
                  (inPic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_run_sim_step<<<l_blocks, l_threads>>>(inPic, inRule, line);

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

__device__ bool elementary_rule(uchar inRule, bool inLeft, bool inCenter, bool inRight)
{
    return (inRule >> (inLeft << 2 | inCenter << 1 | inRight)) & 1;
}



