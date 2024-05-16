#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "cuda_img.h"
#include "uni_mem_allocator.h"

using namespace cv;

void cuda_blur(CudaPic inPic, CudaPic outPic, int neigh);

int main(int argc, char **argv)
{
    UniformAllocator allocator;
    Mat::setDefaultAllocator(&allocator);

    Mat inMat = imread(argv[1], IMREAD_COLOR);
    Mat outMat = Mat(inMat.size(), CV_8UC3);
    Mat outMat2 = Mat(inMat.size(), CV_8UC3);
    Mat outMat3 = Mat(inMat.size(), CV_8UC3);

    CudaPic inPic = CudaPic(inMat);
    CudaPic outPic = CudaPic(outMat);
    CudaPic outPic2 = CudaPic(outMat2);
    CudaPic outPic3 = CudaPic(outMat3);

    cuda_blur(inPic, outPic, 1);
    cuda_blur(inPic, outPic2, 3);
    cuda_blur(inPic, outPic3, 5);

    imshow("IN", inMat);

    imshow("OUT - 1px", outMat);
    imshow("OUT - 3px", outMat2);
    imshow("OUT - 5px", outMat3);

    waitKey(0);

    return 0;
}
