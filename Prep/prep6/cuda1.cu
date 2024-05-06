#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

// Kernel to combine two images next to each other
__global__ void kernel_combine_overlay(CudaPic inPic1, CudaPic inPic2, CudaPic overPic, CudaPic outPic)
{
    // Calculate the x and y coordinates of the thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the image boundaries
    if(x >= outPic.m_size.x) return;
    if(y >= outPic.m_size.y) return;

    if(inPic1.m_size.x == inPic2.m_size.x)
    {
        if(y < inPic1.m_size.y)
        {
            uchar4 l_bgra1 = inPic1.getData<uchar4>(x, y);
            uchar4 l_bgraOver = overPic.getData<uchar4>(x, y);
            l_bgraOver.w = 127;
            uchar4 l_bgra1Over = make_uchar4((l_bgra1.x * (255 - l_bgraOver.w) + l_bgraOver.x * l_bgraOver.w) / 255,
                                            (l_bgra1.y * (255 - l_bgraOver.w) + l_bgraOver.y * l_bgraOver.w) / 255,
                                            (l_bgra1.z * (255 - l_bgraOver.w) + l_bgraOver.z * l_bgraOver.w) / 255,
                                            255);

            outPic.setData<uchar4>(x, y, l_bgra1Over);
        }
        else
        {
            uchar4 l_bgra2 = inPic2.getData<uchar4>(x, y - inPic1.m_size.y);
            uchar4 l_bgraOver = overPic.getData<uchar4>(x, y);
            l_bgraOver.w = 127;
            uchar4 l_bgra2Over = make_uchar4((l_bgra2.x * (255 - l_bgraOver.w) + l_bgraOver.x * l_bgraOver.w) / 255,
                                            (l_bgra2.y * (255 - l_bgraOver.w) + l_bgraOver.y * l_bgraOver.w) / 255,
                                            (l_bgra2.z * (255 - l_bgraOver.w) + l_bgraOver.z * l_bgraOver.w) / 255,
                                            255);

            outPic.setData<uchar4>(x, y, l_bgra2Over);
        }
    }
    else if(inPic1.m_size.y == inPic2.m_size.y)
    {
        if(x < inPic1.m_size.x)
        {
            uchar4 l_bgra1 = inPic1.getData<uchar4>(x, y);
            uchar4 l_bgraOver = overPic.getData<uchar4>(x, y);
            l_bgraOver.w = 127;
            uchar4 l_bgra1Over = make_uchar4((l_bgra1.x * (255 - l_bgraOver.w) + l_bgraOver.x * l_bgraOver.w) / 255,
                                            (l_bgra1.y * (255 - l_bgraOver.w) + l_bgraOver.y * l_bgraOver.w) / 255,
                                            (l_bgra1.z * (255 - l_bgraOver.w) + l_bgraOver.z * l_bgraOver.w) / 255,
                                            255);

            outPic.setData<uchar4>(x, y, l_bgra1Over);
        }
        else
        {
            uchar4 l_bgra2 = inPic2.getData<uchar4>(x - inPic1.m_size.x, y);
            uchar4 l_bgraOver = overPic.getData<uchar4>(x, y);
            l_bgraOver.w = 127;
            uchar4 l_bgra2Over = make_uchar4((l_bgra2.x * (255 - l_bgraOver.w) + l_bgraOver.x * l_bgraOver.w) / 255,
                                            (l_bgra2.y * (255 - l_bgraOver.w) + l_bgraOver.y * l_bgraOver.w) / 255,
                                            (l_bgra2.z * (255 - l_bgraOver.w) + l_bgraOver.z * l_bgraOver.w) / 255,
                                            255);

            outPic.setData<uchar4>(x, y, l_bgra2Over);
        }
    }
}

__global__ void kernel_transparent_overlay(CudaPic inPic, CudaPic inPic2, CudaPic outPic)
{
    // Calculate the x and y coordinates of the thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the image boundaries
    if(x >= inPic.m_size.x) return;
    if(y >= inPic.m_size.y) return;

    uchar4 l_bgra1 = inPic.getData<uchar4>(x, y);
    // uchar4 l_bgra2 = inPic2.getData<uchar4>(x, y);
    // l_bgra2.w = 127;

    // uchar4 l_bgraOut = make_uchar4((l_bgra1.x * (255 - l_bgra2.w) + l_bgra2.x * l_bgra2.w) / 255,
    //                                 (l_bgra1.y * (255 - l_bgra2.w) + l_bgra2.y * l_bgra2.w) / 255,
    //                                 (l_bgra1.z * (255 - l_bgra2.w) + l_bgra2.z * l_bgra2.w) / 255,
    //                                 255);
    uchar4 fin = make_uchar4(l_bgra1.x, l_bgra1.y, l_bgra1.z, 255);

    outPic.setData<uchar4>(x, y, fin);
}

__global__ void kernel_copy(CudaPic inPic, CudaPic outPic)
{
    // Calculate the x and y coordinates of the thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the image boundaries
    if(x >= outPic.m_size.x) return;
    if(y >= outPic.m_size.y) return;

    uchar4 l_bgra = inPic.getData<uchar4>(x, y);
    if(x < 50 && y < 50)
        printf("x: %d, y: %d, r: %d, g: %d, b: %d, a: %d\n", x, y, l_bgra.x, l_bgra.y, l_bgra.z, l_bgra.w);
    //l_bgra.w = 255;

    //outPic.setData<uchar4>(x, y, l_bgra);
}

__global__ void kernel_shrink(CudaPic inPic, CudaPic outPic)
{
    // Calculate the x and y coordinates of the thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the image boundaries
    if(x >= outPic.m_size.x) return;
    if(y >= outPic.m_size.y) return;

    // if(!(x < inPic.m_size.x / 2 && y < inPic.m_size.y / 2))
    // {
    //     outPic.setData<uchar4>(x, y, make_uchar4(0, 0, 0, 255));
    //     return;
    // }

    int l_x = x * 2;
    int l_y = y * 2;

    uchar4 l_bgra = inPic.getData<uchar4>(l_x, l_y);

    outPic.setData<uchar4>(x, y, l_bgra);
}

__global__ void kernel_split6(CudaPic inPic, CudaPic outPic)
{
    // Calculate the x and y coordinates of the thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the image boundaries
    if(x >= outPic.m_size.x) return;
    if(y >= outPic.m_size.y) return;

    int halfY = inPic.m_size.y / 2;
    int thirdX = inPic.m_size.x / 3 + 1;

    uchar4 l_bgra = inPic.getData<uchar4>(x, y);

    if(x < halfY && y < thirdX)
    {
        outPic.setData<uchar4>(halfY - x, y, l_bgra);
    }
    else if(x < halfY && y >= 2 * thirdX)
    {
        outPic.setData<uchar4>(halfY - x, y, l_bgra);
    }
    else if(x >= halfY && y >= thirdX && y < 2 * thirdX)
    {
        outPic.setData<uchar4>(x - halfY, y, l_bgra);
    }
    else
    {
        //outPic.setData<uchar4>(x, y, l_bgra);
    }
    
}

void cuda_combine_overlay(CudaPic inPic1, CudaPic inPic2, CudaPic overPic, CudaPic outPic)
{
    cudaError_t l_cerr;

    int block_size = 32;

    int blockX, blockY;

    if(inPic1.m_size.x == inPic2.m_size.x)
    {
        blockX = (inPic1.m_size.x + block_size - 1) / block_size;
        blockY = (inPic1.m_size.y + inPic2.m_size.y + block_size - 1) / block_size;
    }
    else if(inPic1.m_size.y == inPic2.m_size.y)
    {
        blockX = (inPic1.m_size.x + inPic2.m_size.x + block_size - 1) / block_size;
        blockY = (inPic1.m_size.y + block_size - 1) / block_size;
    }

    dim3 l_blocks(blockX, blockY);

    dim3 l_threads(block_size, block_size);
    
    kernel_combine_overlay<<<l_blocks, l_threads>>>(inPic1, inPic2, overPic, outPic);

    // Check for any CUDA errors
    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    // Synchronize the device
    cudaDeviceSynchronize();
}

void cuda_transparent_overlay(CudaPic inPic, CudaPic inPic2, CudaPic outPic)
{
    cudaError_t l_cerr;

    int l_block_size = 16;

    dim3 l_blocks((inPic.m_size.x + l_block_size - 1) / l_block_size,
                  (inPic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_transparent_overlay<<<l_blocks, l_threads>>>(inPic, inPic2, outPic);

    // Check for any CUDA errors
    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    // Synchronize the device
    cudaDeviceSynchronize();
}

void cuda_shrink(CudaPic inPic, CudaPic outPic)
{
    cudaError_t l_cerr;

    int block_size = 32;

    int blockX = (outPic.m_size.x + block_size - 1) / block_size;
    int blockY = (outPic.m_size.y + block_size - 1) / block_size;

    dim3 l_blocks(blockX, blockY);

    dim3 l_threads(block_size, block_size);

    kernel_shrink<<<l_blocks, l_threads>>>(inPic, outPic);

    // Check for any CUDA errors
    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    // Synchronize the device
    cudaDeviceSynchronize();
}

void cuda_split6(CudaPic inPic, CudaPic outPic)
{
    cudaError_t l_cerr;

    int l_block_size = 32;

    dim3 l_blocks((inPic.m_size.x + l_block_size - 1) / l_block_size,
                  (inPic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_split6<<<l_blocks, l_threads>>>(inPic, outPic);

    // Check for any CUDA errors
    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    // Synchronize the device
    cudaDeviceSynchronize();
}

void cuda_copy(CudaPic inPic, CudaPic outPic)
{
    cudaError_t l_cerr;

    int l_block_size = 32;

    dim3 l_blocks((inPic.m_size.x + l_block_size - 1) / l_block_size,
                  (inPic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_copy<<<l_blocks, l_threads>>>(inPic, outPic);

    // Check for any CUDA errors
    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    // Synchronize the device
    cudaDeviceSynchronize();
}

