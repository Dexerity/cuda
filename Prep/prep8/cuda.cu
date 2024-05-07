#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

__global__ void kernel_mirror(CudaPic inPic, CudaPic outPic, int axis)
{
    // Calculate the x and y coordinates of the thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the image boundaries
    if(x >= inPic.m_size.x) return;
    if(y >= inPic.m_size.y) return;

    uchar3 l_bgra = inPic.getData<uchar3>(x, y);

    if(axis == 0)
    {
        outPic.setData<uchar3>(x, inPic.m_size.y - y - 1, l_bgra);
    }
    else
    {
        outPic.setData<uchar3>(inPic.m_size.x - x - 1, y, l_bgra);
    }
}

__global__ void kernel_darken(CudaPic inPic, CudaPic outPic)
{
    // Calculate the x and y coordinates of the thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the image boundaries
    if(x >= inPic.m_size.x) return;
    if(y >= inPic.m_size.y) return;

    uchar3 l_bgra = inPic.getData<uchar3>(x, y);

    l_bgra = make_uchar3(l_bgra.x * (y / (float)inPic.m_size.y), l_bgra.y * (y / (float)inPic.m_size.y), l_bgra.z * (y / (float)inPic.m_size.y));

    outPic.setData<uchar3>(x, y, l_bgra);
}

__global__ void kernel_double(CudaPic inPic, CudaPic outPic)
{
    // Calculate the x and y coordinates of the thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the image boundaries
    if(x >= inPic.m_size.x) return;
    if(y >= inPic.m_size.y) return;

    uchar3 l_bgra = inPic.getData<uchar3>(x, y);

    outPic.setData<uchar3>(x, y, l_bgra);
    outPic.setData<uchar3>(x + inPic.m_size.x, y, l_bgra);
}

void cuda_mirror(CudaPic inPic, CudaPic outPic, int axis)
{
    cudaError_t l_cerr;

    int block_size = 32;

    dim3 l_blocks((inPic.m_size.x + block_size - 1) / block_size, (inPic.m_size.y + block_size - 1) / block_size);
    dim3 l_threads(block_size, block_size);

    kernel_mirror<<<l_blocks, l_threads>>>(inPic, outPic, axis);

    // Check for any CUDA errors
    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    // Synchronize the device
    cudaDeviceSynchronize();
}


void cuda_darken(CudaPic inPic, CudaPic outPic)
{
    cudaError_t l_cerr;

    int block_size = 32;

    dim3 l_blocks((inPic.m_size.x + block_size - 1) / block_size, (inPic.m_size.y + block_size - 1) / block_size);
    dim3 l_threads(block_size, block_size);

    kernel_darken<<<l_blocks, l_threads>>>(inPic, outPic);

    // Check for any CUDA errors
    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    // Synchronize the device
    cudaDeviceSynchronize();

}

Mat cuda_double(CudaPic inPic)
{
    cudaError_t l_cerr;

    Mat outMat = Mat(inPic.m_size.y, inPic.m_size.x * 2, CV_8UC3);
    CudaPic outPic = CudaPic(outMat);

    int block_size = 32;

    dim3 l_blocks((inPic.m_size.x + block_size - 1) / block_size, (inPic.m_size.y + block_size - 1) / block_size);
    dim3 l_threads(block_size, block_size);

    kernel_double<<<l_blocks, l_threads>>>(inPic, outPic);

    // Check for any CUDA errors
    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    // Synchronize the device
    cudaDeviceSynchronize();

    return outMat;
}