#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

__global__ void kernel_grayscaleCenter(CudaPic t_colorPic, CudaPic t_grayPic) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= t_colorPic.m_size.x) return;
    if(y >= t_colorPic.m_size.y) return;

    uchar3 l_bgr = t_colorPic.getData<uchar3>(x, y);

    if((x >= 30 && x <= t_colorPic.m_size.x - 30) && (y >= 30 && y <= t_colorPic.m_size.y - 30))
    {
        t_grayPic.setData<uchar3>(x, y, make_uchar3(l_bgr.x * 0.11 + l_bgr.y * 0.59 + l_bgr.z * 0.3,
                                                    l_bgr.x * 0.11 + l_bgr.y * 0.59 + l_bgr.z * 0.3,
                                                    l_bgr.x * 0.11 + l_bgr.y * 0.59 + l_bgr.z * 0.3));
    }
    else
        t_grayPic.setData<uchar3>(x, y, l_bgr);
    
}

__global__ void kernel_halveRGB(CudaPic t_colorPic, CudaPic t_darkPic) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= t_colorPic.m_size.x) return;
    if(y >= t_colorPic.m_size.y) return;

    uchar3 l_bgr = t_colorPic.getData<uchar3>(x, y);

    if((x - t_colorPic.m_size.x/2)*(x - t_colorPic.m_size.x/2) + (y - t_colorPic.m_size.y/2)*(y - t_colorPic.m_size.y/2) <= 50*50)
    {
        t_darkPic.setData<uchar3>(x, y, make_uchar3(l_bgr.x / 2, l_bgr.y / 2, l_bgr.z / 2));
    }
    else
        t_darkPic.setData<uchar3>(x, y, l_bgr);


}

__global__ void kernel_multRGB(CudaPic t_colorPic, CudaPic t_multPic) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= t_colorPic.m_size.x) return;
    if(y >= t_colorPic.m_size.y) return;

    uchar3 l_bgr = t_colorPic.getData<uchar3>(x, y);
    uchar3 l_bgr2 = l_bgr;

    if (l_bgr.x * 2 > 255)
        l_bgr2.x = 255;
    else
        l_bgr2.x = l_bgr.x * 2;

    if (l_bgr.y * 2 > 255)
        l_bgr2.y = 255;
    else
        l_bgr2.y = l_bgr.y * 2;

    if (l_bgr.z * 2 > 255)
        l_bgr2.z = 255;
    else
        l_bgr2.z = l_bgr.z * 2;

    t_multPic.setData<uchar3>(x, y, l_bgr2);
        
}

void cuda_grayscaleCenter(CudaPic t_colorPic, CudaPic t_grayPic) {

    cudaError_t l_cerr;

    int l_block_size = 16;

    dim3 l_blocks((t_colorPic.m_size.x + l_block_size - 1) / l_block_size,
                  (t_colorPic.m_size.y + l_block_size - 1) / l_block_size);

    dim3 l_threads(l_block_size, l_block_size);

    kernel_grayscaleCenter<<<l_blocks, l_threads>>>(t_colorPic, t_grayPic);

    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    cudaDeviceSynchronize();
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

void cuda_multRGB(CudaPic t_colorPic, CudaPic t_multPic) 
{

    cudaError_t l_cerr;

    int l_block_size = 16;

    dim3 l_blocks((t_colorPic.m_size.x + l_block_size - 1) / l_block_size,
                  (t_colorPic.m_size.y + l_block_size - 1) / l_block_size);

    dim3 l_threads(l_block_size, l_block_size);

    kernel_multRGB<<<l_blocks, l_threads>>>(t_colorPic, t_multPic);

    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    cudaDeviceSynchronize();
}
