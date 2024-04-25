# Cuda

Preperation and laboratories for CUDA programming

## Table of Contents

- [Structure](#Structure)
- [Functions](#Functions)


## Structure

| Folder        | Description                                |
|---------------|--------------------------------------------|
| Preperation 1 | Simple image modification                  |
| Preperation 2 | Simple grayscale `uchar1` image generation |


## Functions


<details closed><summary>Preperation 1</summary>

| Function                                                                          | Description                                                                                                           |
| --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `__global__ void kernel_grayscaleCenter(CudaPic t_colorPic, CudaPic t_grayPic)`   | Converts a color image to grayscale with a central square region using weighted averaging of color channels.          |
| `__global__ void kernel_halveRGB(CudaPic t_colorPic, CudaPic t_darkPic)`          | Reduces the brightness of a color image by halving the RGB values in a circular region around the image center.       |
| `__global__ void kernel_multRGB(CudaPic t_colorPic, CudaPic t_multPic)`           | Doubles the RGB values of a color image, capping at 255, for each pixel.                                              |
| `void cuda_grayscaleCenter(CudaPic t_colorPic, CudaPic t_grayPic)`                | Host function to convert a color image to grayscale with a central region using CUDA.                                 |
| `void cuda_halveRGB(CudaPic t_colorPic, CudaPic t_darkPic)`                       | Host function to reduce the brightness of a color image by halving the RGB values in a circular region using CUDA.    |
| `void cuda_multRGB(CudaPic t_colorPic, CudaPic t_multPic)`                        | Host function to double the RGB values of a color image, capping at 255, using CUDA.                                  |

</details>

<details closed><summary>Preperation 2</summary>

| Function                                                          | Description                                                                           |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `__global__ void kernel_clear(CudaPic l_cv_in_pic)`               | Clears the image data by setting all pixel values to 0.                               |
| `__global__ void kernel_BW_gradientCircle(CudaPic l_cv_in_pic)`   | Generates a black and white gradient circle centered in the image.                    |
| `__global__ void kernel_BW_gradientHor(CudaPic l_cv_in_pic)`      | Generates a horizontal black and white gradient across the image.                     |
| `__global__ void kernel_BW_gradientVer(CudaPic l_cv_in_pic)`      | Generates a vertical black and white gradient across the image.                       |
| `__global__ void kernel_checkerboard(CudaPic l_cv_in_pic)`        | Creates a black and white checkerboard pattern across the image.                      |
| `void cuda_clear(CudaPic l_cv_in_pic)`                            | Host function to clear the image using CUDA.                                          |
| `void cuda_BW_gradientCircle(CudaPic l_cv_in_pic)`                | Host function to apply a black and white gradient circle using CUDA.                  |
| `void cuda_BW_gradientHor(CudaPic l_cv_in_pic)`                   | Host function to apply a horizontal black and white gradient using CUDA.              |
| `void cuda_BW_gradientVer(CudaPic l_cv_in_pic)`                   | Host function to apply a vertical black and white gradient using CUDA.                |
| `void cuda_checkerboard(CudaPic l_cv_in_pic)`                     | Host function to create a black and white checkerboard pattern using CUDA.            |

</details>
