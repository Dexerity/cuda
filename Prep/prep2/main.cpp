#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "cuda_img.h"
#include "uni_mem_allocator.h"

using namespace cv;

void cuda_clear(CudaPic l_cv_in_pic);
void cuda_BW_gradientCircle(CudaPic l_cv_in_pic);
void cuda_BW_gradientHor(CudaPic l_cv_in_pic);
void cuda_BW_gradientVer(CudaPic l_cv_in_pic);
void cuda_checkerboard(CudaPic l_cv_in_pic);
void cuda_color_checkerboard(CudaPic l_cv_in_pic);

int main(int argc, char **argv)
{
    UniformAllocator allocator;
    Mat::setDefaultAllocator( &allocator );

   
    Mat l_cv_empty_mat(512, 512, CV_8UC1);
    Mat l_cv_empty_mat2(512, 512, CV_8UC3);

    CudaPic pic = CudaPic(l_cv_empty_mat);
    CudaPic pic2 = CudaPic(l_cv_empty_mat2);

    cuda_BW_gradientCircle(pic);
    imshow("Image", l_cv_empty_mat);
    waitKey(0);
    imwrite("BW_gradientCircle.jpg", l_cv_empty_mat);
    cuda_clear(pic);

    cuda_BW_gradientHor(pic);
    imshow("Image", l_cv_empty_mat);
    waitKey(0);
    imwrite("BW_gradientHor.jpg", l_cv_empty_mat);
    cuda_clear(pic);

    cuda_BW_gradientVer(pic);
    imshow("Image", l_cv_empty_mat);
    waitKey(0);
    imwrite("BW_gradientVer.jpg", l_cv_empty_mat);
    cuda_clear(pic);
    
    cuda_checkerboard(pic);
    imshow("Image", l_cv_empty_mat);
    waitKey(0);
    imwrite("checkerboard.jpg", l_cv_empty_mat);
    cuda_clear(pic);

    cuda_color_checkerboard(pic2);
    imshow("Image", l_cv_empty_mat2);
    waitKey(0);
    imwrite("color_checkerboard.jpg", l_cv_empty_mat2);
    cuda_clear(pic2);

    return 0;
}