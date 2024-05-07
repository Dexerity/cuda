#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "cuda_img.h"
#include "uni_mem_allocator.h"

using namespace cv;

void cuda_mirror(CudaPic inPic, CudaPic outPic, int axis);
void cuda_darken(CudaPic inPic, CudaPic outPic);
Mat cuda_double(CudaPic inPic);

int main(int argc, char **argv)
{
    // Set default allocator
    UniformAllocator allocator;
    Mat::setDefaultAllocator( &allocator );

    Mat inMat = imread( argv[1], IMREAD_COLOR);
    Mat outMat = Mat( inMat.size(), CV_8UC3 );
    CudaPic inPic = CudaPic( inMat );
    CudaPic outPic = CudaPic( outMat );

    cuda_mirror(inPic, outPic, 1);

    imshow("IN", inMat);
    imshow("OUT", outMat);

    waitKey(0);
    
    cuda_darken(inPic, outPic);

    imshow("IN", inMat);
    imshow("OUT", outMat);

    waitKey(0);

    Mat outMat2 = cuda_double(inPic);

    imshow("IN", inMat);
    imshow("OUT", outMat2);

    waitKey(0);

    return 0;
}