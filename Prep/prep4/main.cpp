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

    int width = mat1.cols;
    int height = mat1.rows;

    Mat mat1copy(height, width, CV_8UC3);
    Mat mat1copy2(height, width, CV_8UC3);
    Mat mat1edit(height, width, CV_8UC3);
    Mat mat1edit2(height, width, CV_8UC3);
    Mat mat1edit3(height, width, CV_8UC3);
    Mat mat1edit4(height, width, CV_8UC3);

    // Mat mat1copy(mat1.size(), CV_8UC3);
    // Mat mat1copy2(mat1copy.size(), CV_8UC3);
    // Mat mat1edit(mat1copy.size(), CV_8UC3);
    // Mat mat1edit2(mat1copy.size(), CV_8UC3);
    // Mat mat1edit3(mat1copy.size(), CV_8UC3);
    // Mat mat1edit4(mat1copy.size(), CV_8UC3);

    CudaPic pic = CudaPic(mat1);
    CudaPic picCopy = CudaPic(mat1copy);
    CudaPic picCopy2 = CudaPic(mat1copy2);
    CudaPic picEdit = CudaPic(mat1edit);
    CudaPic picEdit2 = CudaPic(mat1edit2);
    CudaPic picEdit3 = CudaPic(mat1edit3);
    CudaPic picEdit4 = CudaPic(mat1edit4);

    uchar3 color = make_uchar3(0, 255, 0);

    cuda_flip(pic, picCopy, 1);
    imshow("Vertical", mat1copy);

    cuda_flip(pic, picCopy2, 2);
    imshow("Horizontal", mat1copy2);

    cuda_color_remove(pic, picEdit, color, 0);
    imshow("0% remove", mat1edit);

    cuda_color_remove(pic, picEdit2, color, 0.25);
    imshow("25% remove", mat1edit2);

    cuda_color_remove(pic, picEdit3, color, 0.50);
    imshow("50% remove", mat1edit3);
    
    cuda_color_remove(pic, picEdit4, color, 1);
    imshow("100% remove", mat1edit4);
    

    waitKey(0);

    return 0;
}