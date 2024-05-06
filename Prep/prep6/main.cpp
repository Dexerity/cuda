#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "cuda_img.h"
#include "uni_mem_allocator.h"

using namespace cv;

// Function declarations
void cuda_combine_overlay(CudaPic inPic1, CudaPic inPic2, CudaPic overPic, CudaPic outPic);
void cuda_shrink(CudaPic inPic, CudaPic outPic);
void cuda_split6(CudaPic inPic, CudaPic outPic);
void cuda_transparent_overlay(CudaPic inPic1, CudaPic inPic2, CudaPic outPic);
void cuda_copy(CudaPic inPic, CudaPic outPic);

int main(int argc, char **argv)
{
    // Set default allocator
    UniformAllocator allocator;
    Mat::setDefaultAllocator( &allocator );

    if ( argc > 1 )
    {
        //read rgba
        Mat l_cv_muj_mat = imread( argv[1], IMREAD_UNCHANGED);
        Mat l_cv_muj_mat2 = imread( argv[2], IMREAD_UNCHANGED);
        //Mat l_cv_muj_mat3 = imread( argv[3], IMREAD_UNCHANGED);
        //Mat l_cv_muj_mat3( l_cv_muj_mat.size(), CV_8UC4);

        CudaPic pic = CudaPic( l_cv_muj_mat );
        CudaPic pic2 = CudaPic( l_cv_muj_mat2 );
        //CudaPic pic3 = CudaPic( l_cv_muj_mat3 );

        /*if(pic.m_size.x == pic2.m_size.x)
        {
            Mat l_cv_muj_mat4( pic.m_size.y + pic2.m_size.y, pic.m_size.x, CV_8UC4 );
            CudaPic pic4 = CudaPic( l_cv_muj_mat4 );
            cuda_combine_overlay(pic, pic2, pic3, pic4);
            imshow("Image3 x", l_cv_muj_mat4);
        }
        else if(pic.m_size.y == pic2.m_size.y)
        {
            Mat l_cv_muj_mat4( pic.m_size.y, pic.m_size.x + pic2.m_size.x, CV_8UC4 );
            CudaPic pic4 = CudaPic( l_cv_muj_mat4 );
            cuda_combine_overlay(pic, pic2, pic3, pic4);
            //imshow("Image3 y", l_cv_muj_mat4);
        }*/

        //shrin
        Mat l_cv_muj_mat4( pic.m_size.y / 2, pic.m_size.x / 2, CV_8UC4 );
        CudaPic pic4 = CudaPic( l_cv_muj_mat4 );
        cuda_shrink(pic, pic4);

        imshow("Image", l_cv_muj_mat);
        //imshow("Image2", l_cv_muj_mat2);
        imshow("Image4", l_cv_muj_mat4);

        // printf("size: %d %d\n", l_cv_muj_mat.rows, l_cv_muj_mat.cols);
        // printf("size: %d %d\n", l_cv_muj_mat2.rows, l_cv_muj_mat2.cols);
        Mat l_cv_muj_mat5( pic.m_size.y, pic.m_size.x, CV_8UC4 );
        CudaPic pic5 = CudaPic( l_cv_muj_mat5 );

        cuda_split6(pic, pic5);
        imshow("Image5", l_cv_muj_mat5);

        Mat l_cv_muj_mat6( pic2.m_size.y, pic2.m_size.x, CV_8UC3 );
        CudaPic pic6 = CudaPic( l_cv_muj_mat6 );
        imshow("test", l_cv_muj_mat2);
        cuda_transparent_overlay(pic2, pic, pic6);

        
        imshow("Image6", l_cv_muj_mat6);

        Mat l_cv_muj_mat7( pic2.m_size.y, pic2.m_size.x, CV_8UC4 );
        CudaPic pic7 = CudaPic( l_cv_muj_mat7 );
        cuda_copy(pic2, pic7);
        imshow("Image7", l_cv_muj_mat7);

        


            
        waitKey(0);
    }

    return 0;
}