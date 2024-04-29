#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "cuda_img.h"
#include "uni_mem_allocator.h"

using namespace cv;

// Function declarations
void cuda_clear(CudaPic l_cv_in_pic);
void cuda_rainbowGradient(CudaPic l_cv_in_pic);

int main(int argc, char **argv)
{
    // Set default allocator
    UniformAllocator allocator;
    Mat::setDefaultAllocator( &allocator );

    // Create empty images of the same size
    Mat l_cv_empty_mat(512, 512, CV_8UC1);
    Mat l_cv_empty_mat2(512, 512, CV_8UC4);

    // Convert input image to CudaPic
    CudaPic pic = CudaPic(l_cv_empty_mat);
    CudaPic pic2 = CudaPic(l_cv_empty_mat2);

    cuda_rainbowGradient(pic2);
    imshow("Rainbow Gradient", l_cv_empty_mat2);
    imwrite("rainbowGradient.png", l_cv_empty_mat2);

    waitKey(0);

    return 0;
}