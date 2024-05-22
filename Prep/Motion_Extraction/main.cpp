#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "cuda_img.h"
#include "stack.h"
#include "uni_mem_allocator.h"

using namespace cv;

Mat motionExtract(Mat frame1, Mat frame2);
void cuda_combineTransparency(CudaPic inPicBG, CudaPic inPicFG, CudaPic outPic);
void cuda_upChannel(CudaPic inPic, CudaPic outPic);
void cuda_downChannel(CudaPic inPic, CudaPic outPic);
void cuda_invert(CudaPic inPic);
void cuda_multRGB(CudaPic t_colorPic, CudaPic t_multPic);


int main(int argc, char **argv)
{
    UniformAllocator allocator;
    Mat::setDefaultAllocator(&allocator);

    int frameDelay = 15;
    
    Stack stack(frameDelay);

    VideoCapture cap(argv[1]);
    VideoWriter video("output.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), cap.get(CAP_PROP_FPS), Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));

    if (!cap.isOpened())
    {
        printf("Error: Could not open video file\n");
        return -1;
    }

    while (1)
    {
        Mat frame;
        
        cap >> frame;

        stack.push(frame);

        Mat tmp(frame.size(), CV_8UC3);

        if (frame.empty())
            break;

        Mat res = motionExtract(frame, stack.peek());

        namedWindow("Frame", WINDOW_NORMAL);
        resizeWindow("Frame", 1600, 900);
        imshow("Frame", res);

        video.write(res);

        if (waitKey(15) >= 0)
            break;
    }

    cap.release();
    video.release();

    return 0;
}

Mat motionExtract(Mat frame1, Mat frame2)
{
    Mat tmp(frame1.size(), CV_8UC4);
    Mat tmp2(frame1.size(), CV_8UC4);
    Mat res(frame1.size(), CV_8UC4);
    Mat res2(frame1.size(), CV_8UC3);

    CudaPic pic1(frame1);
    CudaPic pic2(frame2);
    CudaPic picTmp(tmp);
    CudaPic picTmp2(tmp2);
    CudaPic picRes(res);
    CudaPic picRes2(res2);

    cuda_upChannel(pic1, picTmp);
    cuda_upChannel(pic2, picTmp2);

    cuda_invert(picTmp2);
   
    cuda_combineTransparency(picTmp, picTmp2, picRes);
    cuda_downChannel(picRes, picRes2);

    return res2;
}