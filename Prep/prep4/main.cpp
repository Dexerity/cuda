#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "cuda_img.h"
#include "uni_mem_allocator.h"

using namespace cv;

// Function declarations
void cuda_flip(CudaPic inPic, CudaPic outPic, int dir);
void cuda_color_remove(CudaPic inPic, CudaPic outPic, uchar3 color, double amount);

int main(int argc, char **argv)
{
    // Set default allocator
    UniformAllocator allocator;
    Mat::setDefaultAllocator( &allocator );

    // Create empty images of the same size
    Mat mat1 = imread(argv[1], IMREAD_COLOR);
    Mat mat1copy(mat1.size(), CV_8UC3);
    Mat mat1copy2(mat1.size(), CV_8UC3);
    Mat mat1edit(mat1.size(), CV_8UC3);
    Mat mat1edit2(mat1.size(), CV_8UC3);
    Mat mat1edit3(mat1.size(), CV_8UC3);
    Mat mat1edit4(mat1.size(), CV_8UC3);

    printf("%d, %d", mat1.rows, mat1.cols);

    CudaPic pic = CudaPic(mat1);
    CudaPic picCopy = CudaPic(mat1copy);
    CudaPic picCopy2 = CudaPic(mat1copy2);
    CudaPic picEdit = CudaPic(mat1edit);
    CudaPic picEdit2 = CudaPic(mat1edit2);
    CudaPic picEdit3 = CudaPic(mat1edit3);
    CudaPic picEdit4 = CudaPic(mat1edit4);

    uchar3 color = make_uchar3(255, 255, 0);  

    cuda_flip(pic, picCopy, 1);
    imshow("Vertical", mat1copy);

    // cuda_flip(pic, picCopy2, 2);
    // imshow("Horizontal", mat1copy2);

    // cuda_color_remove(pic, picEdit, color, 0);
    // imshow("0% remove", mat1edit);

    // cuda_color_remove(pic, picEdit2, color, 0.25);
    // imshow("25% remove", mat1edit2);

    // cuda_color_remove(pic, picEdit3, color, 0.50);
    // imshow("50% remove", mat1edit3);

    // cuda_color_remove(pic, picEdit4, color, 1);
    // namedWindow("100% remove", WINDOW_NORMAL);
    // imshow("100% remove", mat1edit4);

    waitKey(0);

    return 0;
}