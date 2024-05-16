#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <string>

#include "cuda_img.h"
#include "uni_mem_allocator.h"

using namespace cv;

void cuda_mult(CudaPic inPic, CudaPic outPic);
void cuda_insertAt(CudaPic picBG, CudaPic picFG, CudaPic res, int xP, int yP);
void cuda_upChannel(CudaPic inPic, CudaPic outPic);

int main(int argc, char **argv)
{
    UniformAllocator allocator;
    Mat::setDefaultAllocator(&allocator);

    Mat inMat = imread(argv[1], IMREAD_UNCHANGED);
    Mat inMat2(inMat.size(), CV_8UC4);
    Mat outMat(inMat.size(), CV_8UC4);

    CudaPic inPic(inMat);
    CudaPic inPic2(inMat2);
    CudaPic outPic(outMat);

    cuda_upChannel(inPic, inPic2);

    cuda_mult(inPic2, outPic);

    imshow("ResultMult", outMat);

    //imwrite("result.jpg", outMat);

    //printf("Channels: %d\n", inMat2.channels());

    int state = 0;
    int yP = 0;

    while(yP < inPic.m_size.x)
    {
        //turn state into string
        std::string stateStr = std::to_string(state);
        
        Mat curMat = imread(stateStr + ".png", IMREAD_UNCHANGED);
        CudaPic curPic(curMat);

        cuda_insertAt(inPic2, curPic, outPic, yP, 200);

        yP += 2;
        state++;
        if(state > 6)
            state = 0;

        //imwrite("res/" + std::to_string(yP) + ".jpg", outMat);
        imshow("Result", outMat);

        waitKey(100);
    }


    return 0;
}
