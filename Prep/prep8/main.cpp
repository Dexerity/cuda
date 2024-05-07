#include <stdio.h> // Standard input/output library
#include <cuda_device_runtime_api.h> // CUDA device runtime API
#include <cuda_runtime.h> // CUDA runtime
#include <opencv2/opencv.hpp> // OpenCV library

#include "cuda_img.h" // Custom CUDA image header
#include "uni_mem_allocator.h" // Custom uniform memory allocator header

using namespace cv; // OpenCV namespace

void cuda_mirror(CudaPic inPic, CudaPic outPic, int axis); // Function declaration for CUDA mirror operation
void cuda_darken(CudaPic inPic, CudaPic outPic); // Function declaration for CUDA darken operation
Mat cuda_double(CudaPic inPic); // Function declaration for CUDA double operation

int main(int argc, char **argv)
{
    // Set default allocator
    UniformAllocator allocator; // Create an instance of the uniform memory allocator
    Mat::setDefaultAllocator( &allocator ); // Set the default allocator for OpenCV's Mat class

    Mat inMat = imread( argv[1], IMREAD_COLOR); // Read input image from command line argument
    Mat outMat = Mat( inMat.size(), CV_8UC3 ); // Create an output image with the same size as the input image
    CudaPic inPic = CudaPic( inMat ); // Create a CUDA image object from the input image
    CudaPic outPic = CudaPic( outMat ); // Create a CUDA image object for the output image

    cuda_mirror(inPic, outPic, 1); // Perform CUDA mirror operation on the input image

    imshow("IN", inMat); // Display the input image
    imshow("OUT", outMat); // Display the output image

    waitKey(0); // Wait for a key press

    cuda_darken(inPic, outPic); // Perform CUDA darken operation on the input image

    imshow("IN", inMat); // Display the input image
    imshow("OUT", outMat); // Display the output image

    waitKey(0); // Wait for a key press

    Mat outMat2 = cuda_double(inPic); // Perform CUDA double operation on the input image

    imshow("IN", inMat); // Display the input image
    imshow("OUT", outMat2); // Display the output image

    waitKey(0); // Wait for a key press

    return 0; // Return 0 to indicate successful execution
}