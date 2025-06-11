import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import indices


def main():
    image = cv2.imread("image/cameraman.png", cv2.IMREAD_GRAYSCALE) # image read as gray scale

    # test code
    cv2.imshow("OpenCV Test", image)

    kernel_size = 5
    sigma = 2
    # filtered_image = GaussianFilter_JY(image, kernel_size, sigma)
    # cv2.imshow(f"Filtered Image (k = {kernel_size}, sigma = {sigma})", filtered_image)
    filtered_image_cv = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    cv2.imshow(f"OpenCV GaussianBlur (k = {kernel_size}, s = {sigma})", filtered_image_cv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def GaussianFilter_JY(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    # 커널 한 변 길이 검증
    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError("Kernel size must be odd.")

    # Gaussian 커널 생성
    radius = kernel_size // 2
    indices = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(indices, indices)
    gaussian_kernel = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))

    # Gaussian 커널 정규화
    gaussian_kernel /= np.sum(gaussian_kernel)

    # 인풋 이미지와 동일한 크기의 빈 이미지 생성
    height, width = image.shape
    output_image = np.zeros_like(image, dtype=np.float64)

    # 이미지 픽셀을 순회하며 convolution(실제로는 correlation) 연산 수행
    for r_out in range(height):
        for c_out in range(width):
            output_pixel_value = 0.0
            for kr_offset in range(-radius, radius + 1):
                for kc_offset in range(-radius, radius + 1):
                    r_in = r_out + kr_offset
                    c_in = c_out + kc_offset

                    if r_in < 0:
                        eff_r_in = 0
                    elif r_in >= height:
                        eff_r_in = height - 1
                    else:
                        eff_r_in = r_in

                    if c_in < 0:
                        eff_c_in = 0
                    elif c_in >= width:
                        eff_c_in = width - 1
                    else:
                        eff_c_in = c_in

                    kernel_value = gaussian_kernel[kr_offset + radius, kc_offset + radius]
                    output_pixel_value += image[eff_r_in, eff_c_in] * kernel_value

            output_image[r_out, c_out] = output_pixel_value

    output_image = np.clip(output_image, 0, 255)
    output_image = output_image.astype(image.dtype)

    return output_image

def GaussianFilter_SJ(image: np.ndarray, width: int, sigma: float) -> np.ndarray:
    center = width//2

    # Create filter size array
    kernel = np.zeros((width, width))

    for i in range(width):
        for j in range(width):
            x = i-center
            y= j-center
            kernel[i,j] = x**2 + y**2

    kernel /= 2*sigma**2
    kernel = np.exp(-kernel)
    kernel /= kernel.sum()

    image_row, image_col = image.shape
    # output_image = np.zeros((image_row, image_col))
    # padding = width//2

    # padded_image = np.zeros((image_row + 2*padding, image_col + 2*padding))
    # padded_image[padding:padded_image.shape[0]-padding,  padding:padded_image.shape[1]-padding] = image

    output_image = np.zeros((image_row, image_col))

    for i in range(image_row):
        for j in range(image_col):
            val = 0.0
            for m in range(-center, center+1):
                for n in range(-center, center+1):
                    x = min(max(i+m, 0), image_row-1)
                    y = min(max(j+n, 0), image_col-1)
                    val += kernel[m+center, n+center] * image[x,y]
            output_image[i,j] = val

    output_image = np.clip(output_image, 0, 255).astype(image.dtype)

    return output_image

def GaussianFilter_SH(image: np.ndarray, r, sigma) -> np.ndarray:
    # make kernel
    kernel_size = 2*r + 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - r
            y = j - r
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    kernel /= np.sum(kernel)

    # replicate padding
    h, w = image.shape
    pad_img = np.zeros((h + 2*r, w + 2*r), dtype=np.float32)

    pad_img[r:r+h, r:r+w] = image
    pad_img[:r, r:r + w] = image[0:1, :]
    pad_img[r + h:, r:r + w] = image[-1:, :]
    pad_img[:, :r] = pad_img[:, r:r + 1]
    pad_img[:, r + w:] = pad_img[:, r + w - 1:r + w]

    # Filtering
    output = np.zeros_like(image, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = pad_img[i:i + kernel_size, j:j + kernel_size]
            output[i, j] = np.sum(region * kernel)

    output = np.clip(output, 0, 255).astype(np.uint8)

    return output

if __name__ == "__main__":
    main()
