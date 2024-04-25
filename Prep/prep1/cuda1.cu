#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

// Kernel to convert color image to grayscale
__global__ void kernel_grayscaleCenter(CudaPic t_colorPic, CudaPic t_grayPic) {
    // Calculate the x and y coordinates of the thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the image boundaries
    if(x >= t_colorPic.m_size.x) return;
    if(y >= t_colorPic.m_size.y) return;

    // Get the color value at the current pixel
    uchar3 l_bgr = t_colorPic.getData<uchar3>(x, y);

    // Check if the pixel is within the center region
    if((x >= 30 && x <= t_colorPic.m_size.x - 30) && (y >= 30 && y <= t_colorPic.m_size.y - 30))
    {
        // Convert the color to grayscale and set it in the grayscale image
        t_grayPic.setData<uchar3>(x, y, make_uchar3(l_bgr.x * 0.11 + l_bgr.y * 0.59 + l_bgr.z * 0.3,
                                                    l_bgr.x * 0.11 + l_bgr.y * 0.59 + l_bgr.z * 0.3,
                                                    l_bgr.x * 0.11 + l_bgr.y * 0.59 + l_bgr.z * 0.3));
    }
    else
        // If the pixel is outside the center region, copy the color as it is
        t_grayPic.setData<uchar3>(x, y, l_bgr);
}

// Kernel to halve the RGB values of pixels within a circular region
__global__ void kernel_halveRGB(CudaPic t_colorPic, CudaPic t_darkPic) {
    // Calculate the x and y coordinates of the thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the image boundaries
    if(x >= t_colorPic.m_size.x) return;
    if(y >= t_colorPic.m_size.y) return;

    // Get the color value at the current pixel
    uchar3 l_bgr = t_colorPic.getData<uchar3>(x, y);

    // Check if the pixel is within the circular region
    if((x - t_colorPic.m_size.x/2)*(x - t_colorPic.m_size.x/2) + (y - t_colorPic.m_size.y/2)*(y - t_colorPic.m_size.y/2) <= 50*50)
    {
        // Halve the RGB values and set it in the darkened image
        t_darkPic.setData<uchar3>(x, y, make_uchar3(l_bgr.x / 2, l_bgr.y / 2, l_bgr.z / 2));
    }
    else
        // If the pixel is outside the circular region, copy the color as it is
        t_darkPic.setData<uchar3>(x, y, l_bgr);
}

// Kernel to multiply the RGB values of pixels by 2
__global__ void kernel_multRGB(CudaPic t_colorPic, CudaPic t_multPic) {
    // Calculate the x and y coordinates of the thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the image boundaries
    if(x >= t_colorPic.m_size.x) return;
    if(y >= t_colorPic.m_size.y) return;

    // Get the color value at the current pixel
    uchar3 l_bgr = t_colorPic.getData<uchar3>(x, y);
    uchar3 l_bgr2 = l_bgr;

    // Multiply the RGB values by 2, clamping the result to 255
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

    // Set the multiplied color in the output image
    t_multPic.setData<uchar3>(x, y, l_bgr2);
}

// Function to convert color image to grayscale using CUDA
void cuda_grayscaleCenter(CudaPic t_colorPic, CudaPic t_grayPic) {

    cudaError_t l_cerr;

    int l_block_size = 16;

    // Calculate the number of blocks and threads per block
    dim3 l_blocks((t_colorPic.m_size.x + l_block_size - 1) / l_block_size,
                  (t_colorPic.m_size.y + l_block_size - 1) / l_block_size);

    dim3 l_threads(l_block_size, l_block_size);

    // Launch the grayscale conversion kernel
    kernel_grayscaleCenter<<<l_blocks, l_threads>>>(t_colorPic, t_grayPic);

    // Check for any CUDA errors
    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    // Synchronize the device
    cudaDeviceSynchronize();
}

// Function to halve the RGB values of pixels within a circular region using CUDA
void cuda_halveRGB(CudaPic t_colorPic, CudaPic t_darkPic) {

    cudaError_t l_cerr;

    int l_block_size = 16;

    // Calculate the number of blocks and threads per block
    dim3 l_blocks((t_colorPic.m_size.x + l_block_size - 1) / l_block_size,
                  (t_colorPic.m_size.y + l_block_size - 1) / l_block_size);

    dim3 l_threads(l_block_size, l_block_size);

    // Launch the halve RGB kernel
    kernel_halveRGB<<<l_blocks, l_threads>>>(t_colorPic, t_darkPic);

    // Check for any CUDA errors
    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    // Synchronize the device
    cudaDeviceSynchronize();
}

// Function to multiply the RGB values of pixels by 2 using CUDA
void cuda_multRGB(CudaPic t_colorPic, CudaPic t_multPic) 
{

    cudaError_t l_cerr;

    int l_block_size = 16;

    // Calculate the number of blocks and threads per block
    dim3 l_blocks((t_colorPic.m_size.x + l_block_size - 1) / l_block_size,
                  (t_colorPic.m_size.y + l_block_size - 1) / l_block_size);

    dim3 l_threads(l_block_size, l_block_size);

    // Launch the multiply RGB kernel
    kernel_multRGB<<<l_blocks, l_threads>>>(t_colorPic, t_multPic);

    // Check for any CUDA errors
    if((l_cerr = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    // Synchronize the device
    cudaDeviceSynchronize();
}
