#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

__global__ void kernel_blur(CudaPic inPic, CudaPic outPic, int neigh)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= inPic.m_size.x) return;
    if(y >= inPic.m_size.y) return;

    uchar3 l_bgr = inPic.getData<uchar3>(x,y);

    int b = 0;
    int g = 0;
    int r = 0;
    int count = 0;

    
    for(int i = -neigh; i <= neigh; i++)
    {
        for(int j = -neigh; j <= neigh; j++)
        {
            int nx = x + i;
            int ny = y + j;
            if(nx >= 0 && nx < inPic.m_size.x && ny >= 0 && ny < inPic.m_size.y)
            {
                b += inPic.getData<uchar3>(nx, ny).x;
                g += inPic.getData<uchar3>(nx, ny).y;
                r += inPic.getData<uchar3>(nx, ny).z;
                count++;
            }
        }
    }

    b /= count;
    g /= count;
    r /= count;

    l_bgr = make_uchar3(b,g,r);


    outPic.setData<uchar3>(x,y, l_bgr);
}

void cuda_blur(CudaPic inPic, CudaPic outPic, int neigh)
{
    cudaError_t l_cerr;

    int block_size = 32;

    dim3 l_blocks((inPic.m_size.x + block_size - 1) / block_size, (inPic.m_size.y + block_size - 1) / block_size);
    dim3 l_threads(block_size, block_size);

    kernel_blur<<<l_blocks, l_threads>>>(inPic, outPic, neigh);

    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    cudaDeviceSynchronize();
}
