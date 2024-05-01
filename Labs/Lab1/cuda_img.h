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
struct CudaPic
{
  public:
    CudaPic(Mat t_cv_mat)
    {
    this->m_size.x = t_cv_mat.size().width;
    this->m_size.y = t_cv_mat.size().height;
    this->m_p_uchar3 = (uchar3 *)t_cv_mat.data;
    }
    
    template <typename T>
    __host__ __device__ T getData(int x, int y);
    template <typename T>
    __host__ __device__ void setData(int x, int y, T t_data);  
  
  uint3 m_size;             // size of picture
  union {
      void   *m_p_void;     // data of picture
      uchar1 *m_p_uchar1;   // data of picture
      uchar3 *m_p_uchar3;   // data of picture
      uchar4 *m_p_uchar4;   // data of picture
  };
};

template<>
__host__ __device__ inline uchar1 CudaPic::getData<uchar1>(int x, int y)
{
    return m_p_uchar1[y * m_size.x + x];
}

template<>
__host__ __device__ inline uchar3 CudaPic::getData<uchar3>(int x, int y)
{
    return m_p_uchar3[y * m_size.x + x];
}

template<>
__host__ __device__ inline uchar4 CudaPic::getData<uchar4>(int x, int y)
{
    return m_p_uchar4[y * m_size.x + x];
}

template<>
__host__ __device__ inline void CudaPic::setData<uchar1>(int x, int y, uchar1 t_data)
{
    m_p_uchar1[y * m_size.x + x] = t_data;
}

template<>
__host__ __device__ inline void CudaPic::setData<uchar3>(int x, int y, uchar3 t_data)
{
    m_p_uchar3[y * m_size.x + x] = t_data;
}

template<>
__host__ __device__ inline void CudaPic::setData<uchar4>(int x, int y, uchar4 t_data)
{
    m_p_uchar4[y * m_size.x + x] = t_data;
}


