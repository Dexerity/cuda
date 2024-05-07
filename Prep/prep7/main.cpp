#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "cuda_img.h"
#include "uni_mem_allocator.h"

using namespace cv;

Mat cuda_combine4(CudaPic inPic, CudaPic overPic);
void cuda_enlarge(CudaPic inPic, CudaPic outPic);
void cuda_shrink(CudaPic inPic, CudaPic outPic);

int main(int argc, char **argv)
{
    // Set default allocator
    UniformAllocator allocator;
    Mat::setDefaultAllocator( &allocator );

    Mat l_cv_muj_mat = imread( argv[1], IMREAD_UNCHANGED);
    Mat l_cv_muj_mat2 = imread( argv[2], IMREAD_UNCHANGED);
    CudaPic pic = CudaPic( l_cv_muj_mat );
    CudaPic pic2 = CudaPic( l_cv_muj_mat2 );

    Mat l_cv_muj_mat4 = cuda_combine4(pic, pic2);
    CudaPic pic4 = CudaPic( l_cv_muj_mat4 );

    imshow("Image", l_cv_muj_mat4);

    Mat l_cv_muj_mat5( pic4.m_size.y * 2, pic4.m_size.x * 2, CV_8UC4 );
    CudaPic pic5 = CudaPic( l_cv_muj_mat5 );
    cuda_enlarge(pic4, pic5);

    imshow("enlarge", l_cv_muj_mat5);
    
    Mat l_cv_muj_mat6( pic4.m_size.y / 2, pic4.m_size.x / 2, CV_8UC4 );
    CudaPic pic6 = CudaPic( l_cv_muj_mat6 );
    cuda_shrink(pic4, pic6);

    imshow("shrink", l_cv_muj_mat6);

    

    waitKey(0);

    return 0;
}