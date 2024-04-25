#include "cuda_img.h"

using namespace cv;

CudaPic::CudaPic(Mat t_cv_mat)
{
    this->m_size.x = t_cv_mat.size().width;
    this->m_size.y = t_cv_mat.size().height;
    this->m_p_uchar3 = (uchar3 *)t_cv_mat.data;
}

template<>
inline uchar1 CudaPic::getData<uchar1>(int x, int y)
{
    return m_p_uchar1[y * m_size.x + x];
}

template<>
inline uchar3 CudaPic::getData<uchar3>(int x, int y)
{
    return m_p_uchar3[y * m_size.x + x];
}

template<>
inline uchar4 CudaPic::getData<uchar4>(int x, int y)
{
    return m_p_uchar4[y * m_size.x + x];
}

template<>
inline void CudaPic::setData<uchar1>(int x, int y, uchar1 t_data)
{
    m_p_uchar1[y * m_size.x + x] = t_data;
}

template<>
inline void CudaPic::setData<uchar3>(int x, int y, uchar3 t_data)
{
    m_p_uchar3[y * m_size.x + x] = t_data;
}

template<>
inline void CudaPic::setData<uchar4>(int x, int y, uchar4 t_data)
{
    m_p_uchar4[y * m_size.x + x] = t_data;
}

