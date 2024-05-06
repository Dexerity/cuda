#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "uni_mem_allocator.h"
#include "cuda_img.h"

using namespace cv;

// imwrite
// imshow
// imread(IM_COLOR)
// waitKey(0)

// blockSize <1, 32>

// CV_8U = uchar1
// CV_8UC3 = uchar3
// CV_8UC4 = uchar4 (with alpha channel)

void concatImages(CudaPic image1, CudaPic image2, CudaPic image3);

int main(int argc, const char** argv) {
    // set default allocator
    UniformAllocator allocator;
    Mat::setDefaultAllocator(&allocator);

    if(argc != 3) {
        printf("Error!");
        return 1;
    }

    Mat mat1 = imread(argv[1], IMREAD_COLOR);
    Mat mat2 = imread(argv[2], IMREAD_COLOR);

    CudaPic image1 = CudaPic(mat1);
    CudaPic image2 = CudaPic(mat2);

    // Height x width
    Mat mat3(image1.m_size.y, image1.m_size.x + image2.m_size.x, CV_8UC3);
    CudaPic image3 = CudaPic(mat3);

    concatImages(image1, image2, image3);

    imshow("Image", mat3);

    waitKey(0);
    return 0;
}