#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

__global__ void kernel_combineTransparency(CudaPic inPicBG, CudaPic inPicFG, CudaPic outPic)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= inPicBG.m_size.x) return;
    if(y >= inPicBG.m_size.y) return;

    uchar4 l_background = inPicBG.getData<uchar4>(x, y);
    uchar4 l_foreground = inPicFG.getData<uchar4>(x, y);

    //overlay the foreground image on the background image
    float l_alpha = l_foreground.w / 255.0f;
    uchar4 l_result;
    l_result.x = l_foreground.x * l_alpha + l_background.x * (1 - l_alpha);
    l_result.y = l_foreground.y * l_alpha + l_background.y * (1 - l_alpha);
    l_result.z = l_foreground.z * l_alpha + l_background.z * (1 - l_alpha);
    l_result.w = 255;

    outPic.setData<uchar4>(x, y, l_result);
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

__global__ void kernel_downChannel(CudaPic inPic, CudaPic outPic)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= inPic.m_size.x) return;
    if(y >= inPic.m_size.y) return;

    uchar4 l_color = inPic.getData<uchar4>(x, y);
    outPic.setData<uchar3>(x, y, make_uchar3(l_color.x, l_color.y, l_color.z));
}

__global__ void kernel_invert(CudaPic inPic)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= inPic.m_size.x) return;
    if(y >= inPic.m_size.y) return;

    uchar4 l_color = inPic.getData<uchar4>(x, y);

    if(l_color.w == 0) return;

    l_color.x = 255 - l_color.x;
    l_color.y = 255 - l_color.y;
    l_color.z = 255 - l_color.z;
    l_color.w = 128;
    inPic.setData<uchar4>(x, y, l_color);
}

void cuda_combineTransparency(CudaPic inPicBG, CudaPic inPicFG, CudaPic outPic)
{
    cudaError_t l_cerr;

    int block_size = 32;

    dim3 l_blocks((inPicBG.m_size.x + block_size - 1) / block_size, (inPicBG.m_size.y + block_size - 1) / block_size);
    dim3 l_threads(block_size, block_size);

    kernel_combineTransparency<<<l_blocks, l_threads>>>(inPicBG, inPicFG, outPic);

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

void cuda_downChannel(CudaPic inPic, CudaPic outPic)
{
    cudaError_t l_cerr;

    int block_size = 32;

    dim3 l_blocks((inPic.m_size.x + block_size - 1) / block_size, (inPic.m_size.y + block_size - 1) / block_size);
    dim3 l_threads(block_size, block_size);

    kernel_downChannel<<<l_blocks, l_threads>>>(inPic, outPic);

    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    cudaDeviceSynchronize();
}

void cuda_invert(CudaPic inPic)
{
    cudaError_t l_cerr;

    int block_size = 16;

    dim3 l_blocks((inPic.m_size.x + block_size - 1) / block_size, (inPic.m_size.y + block_size - 1) / block_size);
    dim3 l_threads(block_size, block_size);

    kernel_invert<<<l_blocks, l_threads>>>(inPic);

    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    cudaDeviceSynchronize();
}
