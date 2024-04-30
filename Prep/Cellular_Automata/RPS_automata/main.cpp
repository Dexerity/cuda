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


int main(int argc, char **argv)
{
    // Set default allocator
    UniformAllocator allocator;
    Mat::setDefaultAllocator( &allocator );

    int step = 0;

    // Create empty images of the same size
    Mat matSource = imread(argv[1], IMREAD_COLOR);
    //Mat matSource(100, 100, CV_8U)
    Mat matNext(matSource.size(), CV_8UC3);

    CudaPic picSource = CudaPic(matSource);
    CudaPic picNext = CudaPic(matNext);

    //cuda_random_canvas(picSource);

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
        memcpy(matSource.data, matNext.data, picSource.m_size.x * picSource.m_size.y * sizeof(uchar3));

        waitKey(0);
    }
    
    

    waitKey(0);

    return 0;
}