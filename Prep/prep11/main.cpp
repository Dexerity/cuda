#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "cuda_img.h"
#include "uni_mem_allocator.h"

using namespace cv;

void cuda_insertAt(CudaPic picBG, CudaPic picFG, CudaPic res, int xP, int yP);

int main(int argc, char **argv)
{
    UniformAllocator allocator;
    Mat::setDefaultAllocator(&allocator);

    int speed = 2;
    bool xDir = true, yDir = true;
    int xP = 100, yP = 100;

    Mat matBG = imread(argv[1], IMREAD_UNCHANGED);
    Mat matFG = imread(argv[2], IMREAD_UNCHANGED);
    Mat matRes(matBG.size(), CV_8UC4);

    CudaPic picBG(matBG);
    CudaPic picFG(matFG);
    CudaPic picRes(matRes);

    while(1)
    {
        cuda_insertAt(picBG, picFG, picRes, xP, yP);

        if(xDir)
        {
            xP += speed;
            if(xP > picBG.m_size.x - picFG.m_size.x)
            {
                xDir = false;
                xP = picBG.m_size.x - picFG.m_size.x;
            }
        }
        else
        {
            xP -= speed;
            if(xP < 0)
            {
                xDir = true;
                xP = 0;
            }
        }

        if(yDir)
        {
            yP += speed;
            if(yP > picBG.m_size.y - picFG.m_size.y)
            {
                yDir = false;
                yP = picBG.m_size.y - picFG.m_size.y;
            }
        }
        else
        {
            yP -= speed;
            if(yP < 0)
            {
                yDir = true;
                yP = 0;
            }
        }

        imshow("Result", matRes);

        if(waitKey(2) == 27)
            break;
    }



    return 0;
}
