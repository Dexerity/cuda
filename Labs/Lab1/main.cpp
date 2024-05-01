#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "cuda_img.h"
#include "uni_mem_allocator.h"

using namespace cv;

void cuda_halveRGB(CudaPic t_colorPic, CudaPic t_darkPic);
void cuda_block_color_sep(CudaPic inPic, CudaPic outPic);
void cuda_block_color_image(CudaPic inPic, CudaPic outPic1, CudaPic outPic2, CudaPic outPic3);

int main(int argc, char **argv)
{
    UniformAllocator allocator;
    Mat::setDefaultAllocator(&allocator);

    if (argc > 1)
    {
        Mat matSource = imread( argv[1], IMREAD_COLOR );
        Mat matOut(matSource.size(), CV_8UC3);
        Mat matOut2(matSource.size(), CV_8UC3);
        Mat matOut3(matSource.size(), CV_8UC3);

        if (!matSource.data)
        {
            printf("File %s cannot be open!\n", argv[1]);
        }
        else
        {
            CudaPic pic = CudaPic(matSource);
            CudaPic picOut = CudaPic(matOut);
            CudaPic picOut2 = CudaPic(matOut2);
            CudaPic picOut3 = CudaPic(matOut3);

            cuda_halveRGB(pic, picOut);
            namedWindow("Halve RGB", WINDOW_NORMAL);
            resizeWindow("Halve RGB", 400, 400);
            imshow("Halve RGB", matOut);

            cuda_block_color_sep(pic, picOut);
            namedWindow("Block Color Seperation", WINDOW_NORMAL);
            resizeWindow("Block Color Seperation", 400, 400);
            imshow("Block Color Seperation", matOut);

            cuda_block_color_image(pic, picOut, picOut2, picOut3);
            namedWindow("Block Color Image 1", WINDOW_NORMAL);
            resizeWindow("Block Color Image 1", 400, 400);
            imshow("Block Color Image 1", matOut);

            namedWindow("Block Color Image 2", WINDOW_NORMAL);
            resizeWindow("Block Color Image 2", 400, 400);
            imshow("Block Color Image 2", matOut2);

            namedWindow("Block Color Image 3", WINDOW_NORMAL);
            resizeWindow("Block Color Image 3", 400, 400);
            imshow("Block Color Image 3", matOut3);
            
            waitKey(0);
        }
    }

    return 0;
}