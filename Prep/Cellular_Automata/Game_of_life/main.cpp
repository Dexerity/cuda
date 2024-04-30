#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <string>

#include "cuda_img.h"
#include "uni_mem_allocator.h"

using namespace cv;

// Function declarations
void cuda_run_sim_step(CudaPic pic, CudaPic picNext);
void cuda_random_canvas(CudaPic pic);
void cuda_create_stable(CudaPic pic, int x, int y);
void cuda_create_glider(CudaPic pic, int x, int y);

int main(int argc, char **argv)
{
    // Set default allocator
    UniformAllocator allocator;
    Mat::setDefaultAllocator( &allocator );

    int step = 0;

    // Create empty images of the same size
    Mat matSource = imread(argv[1], IMREAD_GRAYSCALE);
    //Mat matSource(100, 100, CV_8U)
    Mat matNext(matSource.size(), CV_8U);

    CudaPic picSource = CudaPic(matSource);
    CudaPic picNext = CudaPic(matNext);

    cuda_random_canvas(picSource);

    cuda_create_stable(picSource, 10, 10);
    cuda_create_stable(picSource, 50, 50);
    cuda_create_glider(picSource, 20, 20);

    // Call the function
    while(step < 1000)
    {
        step++;
        std::string windowName = "Current state " + std::to_string(step);

        namedWindow("Step", WINDOW_NORMAL);
        resizeWindow("Step", 800, 800);
        imshow("Step", matSource);

        imwrite("gif_raw/" + std::to_string(step) + ".png", matSource);

        printf("current step: %d\n", step);

        cuda_run_sim_step(picSource, picNext);
        memcpy(matSource.data, matNext.data, 100 * 100 * sizeof(uchar));

        waitKey(100);
    }
    
    

    waitKey(0);

    return 0;
}