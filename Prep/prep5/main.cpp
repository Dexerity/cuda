#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "cuda_img.h"
#include "uni_mem_allocator.h"

using namespace cv;

// Function declarations
void cuda_sep_rotate(CudaPic pic, CudaPic picCopy);
void cuda_remove_color_quadrant(CudaPic inPic, CudaPic outPic);

int main(int argc, char **argv)
{
    // Set default allocator
    UniformAllocator allocator;
    Mat::setDefaultAllocator( &allocator );

    // Create empty images of the same size
    Mat mat1 = imread(argv[1], IMREAD_COLOR);
    Mat mat1copy(mat1.size(), CV_8UC3);
    Mat mat1copy2(mat1.size(), CV_8UC3);

    CudaPic pic = CudaPic(mat1);
    CudaPic picCopy = CudaPic(mat1copy);
    CudaPic picCopy2 = CudaPic(mat1copy2);


    // Call the function
    cuda_sep_rotate(pic, picCopy);
    imshow("Original", mat1);
    imshow("Result", mat1copy);

    // Call the function
    cuda_remove_color_quadrant(picCopy, picCopy2);
    imshow("Result2", mat1copy2);

    

    waitKey(0);

    return 0;
}