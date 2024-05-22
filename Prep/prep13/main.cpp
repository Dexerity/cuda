#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "cuda_img.h"
#include "uni_mem_allocator.h"

using namespace cv;

void cuda_insert(CudaPic picBG, CudaPic picFG, CudaPic res, uchar3 tint);

int main(int argc, char **argv)
{
    UniformAllocator allocator;
    Mat::setDefaultAllocator(&allocator);

    uchar3 tint = {200, 250, 0};

    Mat matBG = imread(argv[1], IMREAD_UNCHANGED);
    Mat matFG = imread(argv[2], IMREAD_UNCHANGED);

    Mat matRes(matBG.size(), CV_8UC4);

    CudaPic picBG(matBG);
    CudaPic picFG(matFG);
    CudaPic picRes(matRes);

    printf("channels: %d\n", matBG.channels());
    printf("channels: %d\n", matFG.channels());

    cuda_insert(picBG, picFG, picRes, tint);

    imshow("Result", matRes);

    waitKey(0);



    return 0;
}
