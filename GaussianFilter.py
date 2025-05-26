import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import indices


def main():
    image = cv2.imread("image/cameraman.png", cv2.IMREAD_GRAYSCALE)

    # test code
    cv2.imshow("OpenCV Test", image)

    kernel_size = 5
    sigma = 2
    filtered_image = GaussianFilter_JY(image, kernel_size, sigma)
    cv2.imshow(f"Filtered Image (k = {kernel_size}, sigma = {sigma})", filtered_image)
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

    # 이미지 패딩
    padded_image = np.pad(image, pad_width=radius, mode='edge')

    # 인풋 이미지와 동일한 크기의 빈 이미지 생성
    height, width = image.shape
    output_image = np.zeros_like(image, dtype=np.float64)

    # 이미지 픽셀 돌면서 convolution(실제로는 correlation) 연산 수행
    for r in range(height):
        for c in range(width):
            image_patch = padded_image[r : r + kernel_size, c : c + kernel_size]
            output_pixel = np.sum(image_patch * gaussian_kernel)
            output_image[r, c] = output_pixel

    output_image = np.clip(output_image, 0, 255)
    output_image = output_image.astype(image.dtype)

    return output_image

if __name__ == "__main__":
    main()
