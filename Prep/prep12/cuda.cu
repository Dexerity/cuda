#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

__global__ void kernel_mult(CudaPic inPic, CudaPic outPic)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= inPic.m_size.x) return;
    if(y >= inPic.m_size.y) return;

    int half_x = inPic.m_size.x / 2;
    int half_y = inPic.m_size.y / 2;

    uchar3 l_color = inPic.getData<uchar3>(x, y);

    outPic.setData<uchar3>(x / 2, y / 2, l_color);
    outPic.setData<uchar3>(x / 2 + half_x, y / 2, l_color);
    outPic.setData<uchar3>(x / 2, y / 2 + half_y, l_color);
    outPic.setData<uchar3>(x / 2 + half_x, y / 2 + half_y, l_color);
}

void cuda_mult(CudaPic inPic, CudaPic outPic)
{
    cudaError_t l_cerr;

    int block_size = 32;

    dim3 l_blocks((inPic.m_size.x + block_size - 1) / block_size, (inPic.m_size.y + block_size - 1) / block_size);
    dim3 l_threads(block_size, block_size);

    kernel_mult<<<l_blocks, l_threads>>>(inPic, outPic);

    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    cudaDeviceSynchronize();
}
