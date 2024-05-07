#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

__global__ void kernel_combine4(CudaPic inPic, CudaPic overPic, CudaPic outPic)
{
    // Calculate the x and y coordinates of the thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the image boundaries
    if(x >= outPic.m_size.x) return;
    if(y >= outPic.m_size.y) return;


    uchar4 l_bgra;

    if(x > inPic.m_size.x && y > inPic.m_size.y)
    {
        l_bgra = inPic.getData<uchar4>(x - inPic.m_size.x, y - inPic.m_size.y);
    }
    else if(x > inPic.m_size.x)
    {
        l_bgra = inPic.getData<uchar4>(x - inPic.m_size.x, y);
    }
    else if(y > inPic.m_size.y)
    {
        l_bgra = inPic.getData<uchar4>(x, y - inPic.m_size.y);
    }
    else
    {
        l_bgra = inPic.getData<uchar4>(x, y);
    }

    

    //overlay across the image
    uchar4 l_bgraOver = overPic.getData<uchar4>(x, y);
    l_bgraOver.w = 127;
    uchar4 l_bgra1Over = make_uchar4((l_bgra.x * (255 - l_bgraOver.w) + l_bgraOver.x * l_bgraOver.w) / 255,
                                    (l_bgra.y * (255 - l_bgraOver.w) + l_bgraOver.y * l_bgraOver.w) / 255,
                                    (l_bgra.z * (255 - l_bgraOver.w) + l_bgraOver.z * l_bgraOver.w) / 255,
                                    255);

    outPic.setData<uchar4>(x, y, l_bgra1Over);
}

__global__ void kernel_enlarge(CudaPic inPic, CudaPic outPic)
{
    // Calculate the x and y coordinates of the thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the image boundaries
    if(x >= outPic.m_size.x) return;
    if(y >= outPic.m_size.y) return;

    uchar4 l_bgra;

    //enlarge the image by doubling the pixels
    l_bgra = inPic.getData<uchar4>(x / 2, y / 2);

    outPic.setData<uchar4>(x, y, l_bgra);
}

__global__ void kernel_shrink(CudaPic inPic, CudaPic outPic)
{
    // Calculate the x and y coordinates of the thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the image boundaries
    if(x >= outPic.m_size.x) return;
    if(y >= outPic.m_size.y) return;

    uchar4 l_bgra;

    //shrink the image by halving the pixels
    l_bgra = inPic.getData<uchar4>(x * 2, y * 2);

    outPic.setData<uchar4>(x, y, l_bgra);
}

Mat cuda_combine4(CudaPic inPic, CudaPic overPic)
{
    cudaError_t l_cerr;

    Mat l_outPic(inPic.m_size.y * 2, inPic.m_size.x * 2, CV_8UC4);
    CudaPic l_outPicC(l_outPic);

    int l_block_size = 16;

    dim3 l_blocks((l_outPicC.m_size.x + l_block_size - 1) / l_block_size, (l_outPicC.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_combine4<<<l_blocks, l_threads>>>(inPic, overPic, l_outPicC);

    // Check for any CUDA errors
    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    // Synchronize the device
    cudaDeviceSynchronize();

    return l_outPic;
}

void cuda_enlarge(CudaPic inPic, CudaPic outPic)
{
    cudaError_t l_cerr;

    int l_block_size = 16;

    dim3 l_blocks((outPic.m_size.x + l_block_size - 1) / l_block_size, (outPic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_enlarge<<<l_blocks, l_threads>>>(inPic, outPic);

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

    int l_block_size = 16;

    dim3 l_blocks((outPic.m_size.x + l_block_size - 1) / l_block_size, (outPic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_shrink<<<l_blocks, l_threads>>>(inPic, outPic);

    // Check for any CUDA errors
    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    // Synchronize the device
    cudaDeviceSynchronize();
}