#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

__global__ void kernel_extract(CudaPic picFG, CudaPic res, int x1, int x2, int y1, int y2)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= picFG.m_size.x) return;
    if(y >= picFG.m_size.y) return;

    if(x2 > picFG.m_size.x) x2 = picFG.m_size.x;
    if(y2 > picFG.m_size.y) y2 = picFG.m_size.y;

    //insert x1 - x2, y1 -y2 of image at x, y, even if half of it is outside
    if(x >= x1 && y >= y1 && x < x2 && y < y2)
    {
        uchar3 l_color = picFG.getData<uchar3>(x, y);
        res.setData<uchar3>(x - x1, y - y1, l_color);
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

__global__ void kernel_bilin_scale(CudaPic inPic, CudaPic outPic)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= outPic.m_size.x) return;
    if(y >= outPic.m_size.y) return;

    float scale_x = inPic.m_size.x - 1;
    float scale_y = inPic.m_size.y - 1;

    scale_x /= outPic.m_size.x;
    scale_y /= outPic.m_size.y;

    int l_resize_x = x;
    int l_resize_y = y;

    float orig_x = l_resize_x * scale_x;
    float orig_y = l_resize_y * scale_y;

    float f_dif_x = orig_x - (int)orig_x;
    float f_dif_y = orig_y - (int)orig_y;

    uchar3 bgr00 = inPic.getData<uchar3>((int)orig_x, (int)orig_y);
    uchar3 bgr01 = inPic.getData<uchar3>((int)orig_x, (int)orig_y + 1);
    uchar3 bgr10 = inPic.getData<uchar3>((int)orig_x + 1, (int)orig_y);
    uchar3 bgr11 = inPic.getData<uchar3>((int)orig_x + 1, (int)orig_y + 1);

    uchar3 bgr;

    bgr.x = (1 - f_dif_x) * (1 - f_dif_y) * bgr00.x + (1 - f_dif_y) * f_dif_x * bgr10.x + f_dif_y * (1 - f_dif_x) * bgr01.x + f_dif_x * f_dif_y * bgr11.x;
    bgr.y = (1 - f_dif_x) * (1 - f_dif_y) * bgr00.y + (1 - f_dif_y) * f_dif_x * bgr10.y + f_dif_y * (1 - f_dif_x) * bgr01.y + f_dif_x * f_dif_y * bgr11.y;
    bgr.z = (1 - f_dif_x) * (1 - f_dif_y) * bgr00.z + (1 - f_dif_y) * f_dif_x * bgr10.z + f_dif_y * (1 - f_dif_x) * bgr01.z + f_dif_x * f_dif_y * bgr11.z;

    outPic.setData<uchar3>(l_resize_x, l_resize_y, bgr);
}

__global__ void kernel_rotate(CudaPic inPic, CudaPic outPic)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= inPic.m_size.x) return;
    if(y >= inPic.m_size.y) return;

    uchar3 l_color = inPic.getData<uchar3>(x, y);

    outPic.setData<uchar3>(y, inPic.m_size.x - x - 1, l_color);
}

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

__global__ void kernel_rot(CudaPic inPic, CudaPic outPic, float alpha)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= inPic.m_size.x) return;
    if(y >= inPic.m_size.y) return;

    float t_sin = sinf(alpha);
    float t_cos = cosf(alpha);

    int l_rotate_x = x;
    int l_rotate_y = y;
    
    int l_crotate_x = l_rotate_x - outPic.m_size.x / 2;
    int l_crotate_y = l_rotate_y - outPic.m_size.y / 2;

    float l_corig_x = t_cos * l_crotate_x - t_sin * l_crotate_y;
    float l_corig_y = t_sin * l_crotate_x + t_cos * l_crotate_y;

    int l_orig_x = l_corig_x + inPic.m_size.x / 2;
    int l_orig_y = l_corig_y + inPic.m_size.y / 2;

    if ( l_orig_x < 0 || l_orig_x >= inPic.m_size.x ) return;
    if ( l_orig_y < 0 || l_orig_y >= inPic.m_size.y ) return;

    outPic.setData<uchar3>(l_rotate_y, l_rotate_x, inPic.getData<uchar3>(l_orig_y, l_orig_x));
    
}

__global__ void kernel_insertAt(CudaPic picBG, CudaPic picFG, CudaPic res, int xP, int yP)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= picBG.m_size.x) return;
    if(y >= picBG.m_size.y) return;

    res.setData<uchar4>(x, y, picBG.getData<uchar4>(x, y));

    //insert image at x, y, even if half of it is outside
    if(x >= xP && y >= yP && x < xP + picFG.m_size.x && y < yP + picFG.m_size.y)
    {
        uchar3 l_color = picFG.getData<uchar3>(x - xP, y - yP);
        res.setData<uchar3>(x, y, l_color);
    }
}

Mat cuda_extract(CudaPic picFG, int x1, int x2, int y1, int y2)
{
    Mat resMat(y2 - y1, x2 - x1, CV_8UC3);
    CudaPic res(resMat);

    cudaError_t l_cerr;

    int block_size = 32;

    dim3 l_blocks((picFG.m_size.x + block_size - 1) / block_size, (picFG.m_size.y + block_size - 1) / block_size);
    dim3 l_threads(block_size, block_size);

    kernel_extract<<<l_blocks, l_threads>>>(picFG, res, x1, x2, y1, y2);

    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    cudaDeviceSynchronize();

    return resMat;
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

void cuda_bilin_scale(CudaPic picSrc, CudaPic picDst)
{
    cudaError_t l_cerr;

    int block_size = 32;

    dim3 l_blocks((picDst.m_size.x + block_size - 1) / block_size, (picDst.m_size.y + block_size - 1) / block_size);
    dim3 l_threads(block_size, block_size);

    kernel_bilin_scale<<<l_blocks, l_threads>>>(picSrc, picDst);

    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    cudaDeviceSynchronize();
}

void cuda_rotate(CudaPic picSrc, CudaPic picDst)
{
    cudaError_t l_cerr;

    int block_size = 32;

    dim3 l_blocks((picDst.m_size.x + block_size - 1) / block_size, (picDst.m_size.y + block_size - 1) / block_size);
    dim3 l_threads(block_size, block_size);

    kernel_rotate<<<l_blocks, l_threads>>>(picSrc, picDst);

    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    cudaDeviceSynchronize();
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

void cuda_rot(CudaPic inPic, CudaPic outPic, float alpha)
{
    cudaError_t l_cerr;

    int block_size = 32;

    dim3 l_blocks((outPic.m_size.x + block_size - 1) / block_size, (outPic.m_size.y + block_size - 1) / block_size);
    dim3 l_threads(block_size, block_size);

    kernel_rot<<<l_blocks, l_threads>>>(inPic, outPic, alpha);

    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    cudaDeviceSynchronize();
}

void cuda_insertAt(CudaPic picBG, CudaPic picFG, CudaPic res, int xP, int yP)
{
    cudaError_t l_cerr;

    int block_size = 32;

    dim3 l_blocks((picBG.m_size.x + block_size - 1) / block_size, (picBG.m_size.y + block_size - 1) / block_size);
    dim3 l_threads(block_size, block_size);

    kernel_insertAt<<<l_blocks, l_threads>>>(picBG, picFG, res, xP, yP);

    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    cudaDeviceSynchronize();
}