// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage.
//
// Image interface for CUDA
//
// ***********************************************************************

#pragma once

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/core/mat.hpp>

using namespace cv;

// Structure definition for exchanging data between Host and Device
class CudaPic
{
  public:
    CudaPic(Mat t_cv_mat);
    
    template <typename T>
    T getData(int x, int y);
    template <typename T>
    void setData(int x, int y, T t_data);  
  
  uint3 m_size;             // size of picture
  union {
      void   *m_p_void;     // data of picture
      uchar1 *m_p_uchar1;   // data of picture
      uchar3 *m_p_uchar3;   // data of picture
      uchar4 *m_p_uchar4;   // data of picture
  };
};
