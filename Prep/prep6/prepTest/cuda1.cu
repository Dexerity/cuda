#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

// (uchar3) {255, 255, 255} = make_uchar3(255, 255, 255)

__global__ void kernel_concatImages(CudaPic image1, CudaPic image2, CudaPic image3) {
    // Calculate the x and y coordinates of the thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // int width = image1.size.x;
    // int height = image1.size.y + image2.size.y;

    // int width = image1.size.x + image2.size.x;
    // int height = image1.size.y;

    int width = image1.m_size.x + image2.m_size.x;
    int height = image1.m_size.y;

    // Check if the thread is outside the image boundaries
    if(x >= width) return;
    if(y >= height) return;

    // if (x < image1.m_size.x) {
    //     // First image
    //     image3.setPixel3(x, y, image1.pixelAt3(x, y));
    // } else {
    //     // Second image
    //     image3.setPixel3(x, y, image2.pixelAt3(x - image1.size.x, y));
    // }

    if (x < image1.m_size.x) {
        // First image
        image3.setData<uchar3>(x, y, image1.getData<uchar3>(x, y));
    } else {
        // Second image
        image3.setData<uchar3>(x, y, image2.getData<uchar3>(x - image1.m_size.x, y));
    }
}

void concatImages(CudaPic image1, CudaPic image2, CudaPic image3) {
    cudaError_t err;

    // Set the block size
    int blockSize = 32;

    // Calculate the number of blocks and threads
    int blocksX = (image1.m_size.x + image2.m_size.x + blockSize - 1) / blockSize;
    int blocksY = (image1.m_size.y + blockSize - 1) / blockSize;
    dim3 blocks(blocksX, blocksY);
    dim3 threads(blockSize, blockSize);

    // Launch the kernel
    kernel_concatImages<<<blocks, threads>>>(image1, image2, image3);

    // Check for any CUDA errors
    if((err = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA error [%d]: %s\n", __LINE__, cudaGetErrorString(err));
    }

    // Synchronize the device
    cudaDeviceSynchronize();
}