#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

__global__ void kernel_halveRGB(CudaPic t_colorPic, CudaPic t_darkPic) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= t_colorPic.m_size.x) return;
    if(y >= t_colorPic.m_size.y) return;

    uchar3 l_bgr = t_colorPic.getData<uchar3>(x, y);

    t_darkPic.setData<uchar3>(x, y, make_uchar3(l_bgr.x / 2, l_bgr.y / 2, l_bgr.z / 2));
}

__global__ void kernel_block_color_sep(CudaPic inPic, CudaPic outPic)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= inPic.m_size.x) return;
    if(y >= inPic.m_size.y) return;

    uchar3 l_bgr = inPic.getData<uchar3>(x, y);

    switch((blockIdx.x + blockIdx.y) % 3)
    {
        case 0:
            outPic.setData<uchar3>(x, y, make_uchar3(l_bgr.x, 0, 0));
            break;
        case 1:
            outPic.setData<uchar3>(x, y, make_uchar3(0, l_bgr.y, 0));
            break;
        case 2:
            outPic.setData<uchar3>(x, y, make_uchar3(0, 0, l_bgr.z));
            break;
    }
}

__global__ void kernel_block_color_image(CudaPic inPic, CudaPic outPic1, CudaPic outPic2, CudaPic outPic3)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= inPic.m_size.x) return;
    if(y >= inPic.m_size.y) return;

    uchar3 l_bgr = inPic.getData<uchar3>(x, y);

    switch((blockIdx.x + blockIdx.y + 2) % 3)
    {
        case 0:
            outPic1.setData<uchar3>(x, y, make_uchar3(l_bgr.x, 0, 0));
            break;
        case 1:
            outPic1.setData<uchar3>(x, y, make_uchar3(0, l_bgr.y, 0));
            break;
        case 2:
            outPic1.setData<uchar3>(x, y, make_uchar3(0, 0, l_bgr.z));
            break;
    }

    switch((blockIdx.x + blockIdx.y + 1) % 3)
    {
        case 0:
            outPic2.setData<uchar3>(x, y, make_uchar3(l_bgr.x, 0, 0));
            break;
        case 1:
            outPic2.setData<uchar3>(x, y, make_uchar3(0, l_bgr.y, 0));
            break;
        case 2:
            outPic2.setData<uchar3>(x, y, make_uchar3(0, 0, l_bgr.z));
            break;
    }

    switch((blockIdx.x + blockIdx.y) % 3)
    {
        case 0:
            outPic3.setData<uchar3>(x, y, make_uchar3(l_bgr.x, 0, 0));
            break;
        case 1:
            outPic3.setData<uchar3>(x, y, make_uchar3(0, l_bgr.y, 0));
            break;
        case 2:
            outPic3.setData<uchar3>(x, y, make_uchar3(0, 0, l_bgr.z));
            break;
    }
}

void cuda_halveRGB(CudaPic t_colorPic, CudaPic t_darkPic) {

    cudaError_t l_cerr;

    int l_block_size = 16;

    dim3 l_blocks((t_colorPic.m_size.x + l_block_size - 1) / l_block_size,
                  (t_colorPic.m_size.y + l_block_size - 1) / l_block_size);

    dim3 l_threads(l_block_size, l_block_size);

    kernel_halveRGB<<<l_blocks, l_threads>>>(t_colorPic, t_darkPic);

    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    cudaDeviceSynchronize();
}

void cuda_block_color_sep(CudaPic inPic, CudaPic outPic)
{
    cudaError_t l_cerr;

    int l_block_size = 16;

    dim3 l_blocks((inPic.m_size.x + l_block_size - 1) / l_block_size,
                  (inPic.m_size.y + l_block_size - 1) / l_block_size);

    dim3 l_threads(l_block_size, l_block_size);

    kernel_block_color_sep<<<l_blocks, l_threads>>>(inPic, outPic);

    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    cudaDeviceSynchronize();
}

void cuda_block_color_image(CudaPic inPic, CudaPic outPic1, CudaPic outPic2, CudaPic outPic3)
{
    cudaError_t l_cerr;

    int l_block_size = 16;

    dim3 l_blocks((inPic.m_size.x + l_block_size - 1) / l_block_size,
                  (inPic.m_size.y + l_block_size - 1) / l_block_size);

    dim3 l_threads(l_block_size, l_block_size);

    kernel_block_color_image<<<l_blocks, l_threads>>>(inPic, outPic1, outPic2, outPic3);

    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    cudaDeviceSynchronize();
}