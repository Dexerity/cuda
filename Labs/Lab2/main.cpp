#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "cuda_img.h"
#include "uni_mem_allocator.h"

using namespace cv;

Mat cuda_extract(CudaPic picFG, int x1, int x2, int y1, int y2);
void cuda_insertAt(CudaPic picBG, CudaPic picFG, CudaPic res, int xP, int yP);
void cuda_bilin_scale(CudaPic picSrc, CudaPic picDst);
void cuda_rotate(CudaPic picSrc, CudaPic picDst);
void cuda_blur(CudaPic inPic, CudaPic outPic, int neigh);
void cuda_rot(CudaPic inPic, CudaPic outPic, float alpha);

int main(int argc, char **argv)
{
    UniformAllocator allocator;
    Mat::setDefaultAllocator(&allocator);

    int speed = 2;
    bool xDir = true, yDir = true;
    int xP = 100, yP = 100;

    Mat matBG = imread(argv[1], IMREAD_COLOR);
    Mat matFG = imread(argv[2], IMREAD_COLOR);
    //Mat matRes(matFG.size(), CV_8UC4);
    Mat matRes2(500, 500, CV_8UC3);
    Mat matRes3(500, 500, CV_8UC3);
    Mat matRes4(500, 500, CV_8UC3);
    Mat matRes5(500, 500, CV_8UC3);
    Mat matRes6(matBG.size(), CV_8UC3);

    CudaPic picBG(matBG);
    CudaPic picFG(matFG);
    //CudaPic picRes(matRes);
    CudaPic picRes2(matRes2);
    CudaPic picRes3(matRes3);
    CudaPic picRes4(matRes4);
    CudaPic picRes5(matRes5);
    CudaPic picRes6(matRes6);

    Mat matRes = cuda_extract(picFG, 10, 140, 10, 140);
    CudaPic picRes(matRes);

    cuda_bilin_scale(picRes, picRes2);

    cuda_rotate(picRes2, picRes3);

    cuda_blur(picRes3, picRes4, 2);

    cuda_rot(picRes4, picRes5, 0.5);

    cuda_insertAt(picBG, picRes5, picRes6, 100, 100);

    //imshow("Result", matRes);
    imwrite("result.jpg", matRes);
    imwrite("result2.jpg", matRes2);
    imwrite("result3.jpg", matRes3);
    imwrite("result4.jpg", matRes4);
    imwrite("result5.jpg", matRes5);
    imwrite("result6.jpg", matRes6);
    
    //waitKey(0);


    return 0;
}
