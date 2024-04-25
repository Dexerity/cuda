#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "cuda_img.h"
#include "uni_mem_allocator.h"

using namespace cv;

void cuda_grayscaleCenter(CudaPic t_colorPic, CudaPic t_grayPic);
void cuda_halveRGB(CudaPic t_colorPic, CudaPic t_darkPic);
void cuda_multRGB(CudaPic t_colorPic, CudaPic t_multPic);

int main(int argc, char **argv)
{
    UniformAllocator allocator;
    Mat::setDefaultAllocator( &allocator );

    uint l_x = 50;
    uint l_y = 100;

    uchar3 l_bgr = { 0, 0, 0 };
    uchar3 l_tmp;

    

    if ( argc > 1 )
    {
        Mat l_cv_muj_mat = imread( argv[1], IMREAD_COLOR );
        Mat l_cv_muj_matBW( l_cv_muj_mat.size(), CV_8UC3 );
        Mat l_cv_muj_matDK( l_cv_muj_mat.size(), CV_8UC3 );
        Mat l_cv_muj_matMT( l_cv_muj_mat.size(), CV_8UC3 );

        if ( !l_cv_muj_mat.data )
        {
            printf("File %s cannot be open!\n", argv[1]);
        }
        else
        {
            CudaPic pic = CudaPic( l_cv_muj_mat );
            CudaPic picBW = CudaPic( l_cv_muj_matBW );
            CudaPic picDK = CudaPic( l_cv_muj_matDK );
            CudaPic picMT = CudaPic( l_cv_muj_matMT );

            

            l_tmp = pic.m_p_uchar3[l_y * pic.m_size.x + l_x];

            printf("l_tmp = %d %d %d\n", l_tmp.x, l_tmp.y, l_tmp.z);

            imshow("Image", l_cv_muj_mat); 

            cuda_grayscaleCenter(pic, picBW);

            imshow("ImageBW", l_cv_muj_matBW);

            cuda_halveRGB(pic, picDK);

            imshow("ImageDK", l_cv_muj_matDK);

            cuda_multRGB(pic, picMT);

            imshow("ImageMT", l_cv_muj_matMT);

            l_tmp = picMT.m_p_uchar3[l_y * pic.m_size.x + l_x];

            printf("l_tmp = %d %d %d\n", l_tmp.x, l_tmp.y, l_tmp.z);

            imwrite("output.png", l_cv_muj_matBW);
            imwrite("outputDK.png", l_cv_muj_matDK);
            imwrite("outputMT.png", l_cv_muj_matMT);

            waitKey(0);
        }
    }
        


    

    

    


    return 0;
}