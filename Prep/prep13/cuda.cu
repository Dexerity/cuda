#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

__global__ void kernel_insert(CudaPic picBG, CudaPic picFG, CudaPic res)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= picBG.m_size.x) return;
    if(y >= picBG.m_size.y) return;

    uchar4 l_background = picBG.getData<uchar4>(x, y);
    uchar4 l_foreground = picFG.getData<uchar4>(x, y);

    if(l_background.w == 0) {
        res.setData<uchar4>(x, y, l_foreground);
    } else {
        res.setData<uchar4>(x, y, l_background);
    }
}

__global__ void kernel_upChannel(CudaPic inPic, CudaPic outPic)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= inPic.m_size.x) return;
    if(y >= inPic.m_size.y) return;

    uchar3 l_color = inPic.getData<uchar3>(x, y);
    outPic.setData<uchar4>(x, y, make_uchar4(l_color.x, l_color.y, l_color.z, 255));
}

void cuda_insert(CudaPic picBG, CudaPic picFG, CudaPic res)
{
    cudaError_t l_cerr;

    int block_size = 32;

    dim3 l_blocks((picBG.m_size.x + block_size - 1) / block_size, (picBG.m_size.y + block_size - 1) / block_size);
    dim3 l_threads(block_size, block_size);

    kernel_insert<<<l_blocks, l_threads>>>(picBG, picFG, res);

    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    cudaDeviceSynchronize();
}

void cuda_upChannel(CudaPic inPic, CudaPic outPic)
{
    cudaError_t l_cerr;

    int block_size = 32;

    dim3 l_blocks((inPic.m_size.x + block_size - 1) / block_size, (inPic.m_size.y + block_size - 1) / block_size);
    dim3 l_threads(block_size, block_size);

    kernel_upChannel<<<l_blocks, l_threads>>>(inPic, outPic);

    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    cudaDeviceSynchronize();
}
