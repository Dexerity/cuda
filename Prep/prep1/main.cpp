#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "cuda_img.h"
#include "uni_mem_allocator.h"

using namespace cv;

// Function declarations
void cuda_grayscaleCenter(CudaPic t_colorPic, CudaPic t_grayPic);
void cuda_halveRGB(CudaPic t_colorPic, CudaPic t_darkPic);
void cuda_multRGB(CudaPic t_colorPic, CudaPic t_multPic);

int main(int argc, char **argv)
{
    // Set default allocator
    UniformAllocator allocator;
    Mat::setDefaultAllocator( &allocator );

    if ( argc > 1 )
    {
        // Read input image
        Mat l_cv_muj_mat = imread( argv[1], IMREAD_COLOR );

        // Create empty images of the same size
        Mat l_cv_muj_matBW( l_cv_muj_mat.size(), CV_8UC3 );
        Mat l_cv_muj_matDK( l_cv_muj_mat.size(), CV_8UC3 );
        Mat l_cv_muj_matMT( l_cv_muj_mat.size(), CV_8UC3 );

        if ( !l_cv_muj_mat.data )
        {
            printf("File %s cannot be open!\n", argv[1]);
        }
        else
        {
            // Convert input image to CudaPic
            CudaPic pic = CudaPic( l_cv_muj_mat );
            CudaPic picBW = CudaPic( l_cv_muj_matBW );
            CudaPic picDK = CudaPic( l_cv_muj_matDK );
            CudaPic picMT = CudaPic( l_cv_muj_matMT );

            // Display original image
            imshow("Image", l_cv_muj_mat); 

            // Apply grayscale transformation
            cuda_grayscaleCenter(pic, picBW);

            // Display grayscale image
            imshow("ImageBW", l_cv_muj_matBW);

            // Apply halve RGB transformation
            cuda_halveRGB(pic, picDK);

            // Display halved RGB image
            imshow("ImageDK", l_cv_muj_matDK);

            // Apply multiply RGB transformation
            cuda_multRGB(pic, picMT);

            // Display multiplied RGB image
            imshow("ImageMT", l_cv_muj_matMT);

            // Save transformed images
            imwrite("output.png", l_cv_muj_matBW);
            imwrite("outputDK.png", l_cv_muj_matDK);
            imwrite("outputMT.png", l_cv_muj_matMT);

            // Wait for key press to exit
            waitKey(0);
        }
    }

    return 0;
}