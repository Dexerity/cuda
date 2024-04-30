#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <string>

#include "cuda_img.h"
#include "uni_mem_allocator.h"

using namespace cv;

// Function declarations
void cuda_run_sim_step(CudaPic inPic, uchar inRule, int line);
bool elementary_rule(uchar inRule, bool inLeft, bool inCenter, bool inRight);
void cuda_random_line(CudaPic inPic);

int main(int argc, char **argv)
{
    // Set default allocator
    UniformAllocator allocator;
    Mat::setDefaultAllocator( &allocator );

    int step = 0;
    uchar ruleIn = 0;
    int size = 400;

    // Create empty images of the same size
    // Mat matSource(size, size, CV_8U);
    // Mat matNext(matSource.size(), CV_8U);

    // CudaPic picSource = CudaPic(matSource);
    // CudaPic picNext = CudaPic(matNext);

    // picSource.setData<uchar1>(picSource.m_size.x / 2, 0, {255});

    // cuda_random_line(picSource);

    while(ruleIn < 255)
    {
        step = 0;
        Mat matSource(size, size, CV_8U);
        CudaPic picSource = CudaPic(matSource);
        cuda_random_line(picSource);

        printf("Rule: %d\n", ruleIn);
        while(step < picSource.m_size.y - 1)
        {
            step++;
            cuda_run_sim_step(picSource, ruleIn, step);
        }

        namedWindow("Step", WINDOW_NORMAL);
        resizeWindow("Step", 800, 800);
        imshow("Step", matSource);

        imwrite("rules/" + std::to_string(ruleIn) + ".png", matSource);
        waitKey(0);
        ruleIn++;
    }
    // Call the function
    
    
    

    waitKey(0);

    return 0;
}