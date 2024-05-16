#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "cuda_img.h"
#include "uni_mem_allocator.h"

using namespace cv;

void cuda_mult(CudaPic inPic, CudaPic outPic);

int main(int argc, char **argv)
{
    UniformAllocator allocator;
    Mat::setDefaultAllocator(&allocator);

    Mat inMat = imread(argv[1], IMREAD_COLOR);
    Mat outMat = Mat(inMat.size(), CV_8UC3);

    CudaPic inPic(inMat);
    CudaPic outPic(outMat);

    cuda_mult(inPic, outPic);

    imwrite("result.jpg", outMat);



    return 0;
}
