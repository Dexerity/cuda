# Cuda

Preperation and laboratories for CUDA programming

## Table of Contents

- [Structure](#Structure)
- [Functions](#Functions)


## Structure

| Folder                                                                    | Description                                |
|---------------------------------------------------------------------------|--------------------------------------------|
| [Cellular Automata](https://github.com/Dexerity/cuda/blob/master/Prep/Cellular_Automata/) | Cellular automata simulation |
| [Preperation 1](https://github.com/Dexerity/cuda/blob/master/Prep/prep1/) | Simple image modification                  |
| [Preperation 2](https://github.com/Dexerity/cuda/blob/master/Prep/prep2/) | Simple grayscale `uchar1` image generation |
| [Preperation 3](https://github.com/Dexerity/cuda/blob/master/Prep/prep3/) | Rainbow + alpha gradient `uchar4`               |
| [Preperation 4](https://github.com/Dexerity/cuda/blob/master/Prep/prep4/) | More simple image modifications |
| [Preperation 5](https://github.com/Dexerity/cuda/blob/master/Prep/prep5/) | Image quadrant modification |


## Functions


<details closed><summary>Preperation 1</summary>v

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

<details closed><summary>Preperation 3</summary>

| Function                                                                          | Description                                                                                                           |
| --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `__global__ void kernel_clear(CudaPic l_cv_in_pic)`                               | Clears an image by setting all pixel values to 0. Each thread handles one pixel of the image.                         |
| `__global__ void kernel_rainbowGradient(CudaPic l_cv_in_pic)`                     | Applies a rainbow gradient to an image. Each thread handles one pixel of the image.                                   |
| `void cuda_clear(CudaPic l_cv_in_pic)`                                            | Host function that launches the `kernel_clear` CUDA kernel to clear an image.                                         |
| `void cuda_rainbowGradient(CudaPic l_cv_in_pic)`                                  | Host function that launches the `kernel_rainbowGradient` CUDA kernel to apply a rainbow gradient to an image.         |

</details>

<details closed><summary>Preperation 4</summary>

| Function                                                                          | Description                                                                                                           |
| --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `__global__ void kernel_flip(CudaPic inPic, CudaPic outPic, int dir)`             | Flips an image horizontally or vertically based on the `dir` parameter. Each thread handles one pixel of the image.   |
| `__global__ void kernel_color_remove(CudaPic inPic, CudaPic outPic, uchar3 color, double amount)` | Removes a certain amount of a specific color from an image. Each thread handles one pixel of the image. |
| `void cuda_flip(CudaPic inPic, CudaPic outPic, int dir)`                          | Host function that launches the `kernel_flip` CUDA kernel to flip an image horizontally or vertically.                |
| `void cuda_color_remove(CudaPic inPic, CudaPic outPic, uchar3 color, double amount)` | Host function that launches the `kernel_color_remove` CUDA kernel to remove a certain amount of a specific color from an image. |
</details>

<details closed><summary>Preperation 5</summary>

| Function                                                                          | Description                                                                                                           |
| --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `__global__ void kernel_sep_rotate(CudaPic inPic, CudaPic outPic)`                | Rotates an image in four different directions based on the quadrant. Each thread handles one pixel of the image.      |
| `__global__ void kernel_remove_color_quadrant(CudaPic inPic, CudaPic outPic)`     | Removes a specific color channel from each quadrant of an image. Each thread handles one pixel of the image.          |
| `void cuda_sep_rotate(CudaPic inPic, CudaPic outPic)`                             | Host function that launches the `kernel_sep_rotate` CUDA kernel to rotate an image.                                   |
| `void cuda_remove_color_quadrant(CudaPic inPic, CudaPic outPic)`                  | Host function that launches the `kernel_remove_color_quadrant` CUDA kernel to remove a specific color channel from each quadrant of an image. |
</details>

