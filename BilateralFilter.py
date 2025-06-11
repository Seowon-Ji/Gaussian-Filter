import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import indices


def main():
    # image read as gray scale
    image = cv2.imread("image/cameraman.png", cv2.IMREAD_GRAYSCALE) # (height, width)

    # test code
    # cv2.imshow("OpenCV Test", image)

    kernel_size = 5
    sigmaColor = 25
    sigmaSpace = 5

    filtered_image_cv = BilateralFilter_SJ(image, kernel_size, sigmaColor, sigmaSpace)
    cv2.imshow(f"OpenCV BilaterFilter (k = {kernel_size}, sigma_color = {sigmaColor}, sigma_space = {sigmaSpace})", filtered_image_cv)
    cv2.imshow("Origin image", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_gaussian_function(x: float, y:float, sigma: float) -> float:
    return np.exp(((x**2 + y**2)/sigma**2) * (-0.5))

def BilateralFilter_SJ(image: np.ndarray, width: int, sigmaColor: float, sigmaSpace: float) -> np.ndarray:
    center = width//2

    # Create filter size array
    spatial_kernel = np.zeros((width, width))

    for i in range(width):
        for j in range(width):
            x = i-center
            y = j-center
            
            spatial_kernel[i,j] = get_gaussian_function(x, y, sigmaSpace)

    image_row, image_col = image.shape
    output_image = np.zeros((image_row, image_col))

    for i in range(image_row):
        for j in range(image_col):
            val = 0.0
            weight_sum = 0.0
            center_val = image[i,j]

            for m in range(-center, center+1):
                for n in range(-center, center+1):
                    x = min(max(i+m, 0), image_row-1)
                    y = min(max(j+n, 0), image_col-1)

                    color_diff = float(image[x,y]) - float(center_val)

                    color_similarity = get_gaussian_function(color_diff, 0, sigmaColor)
                    weight = color_similarity*spatial_kernel[m+center, n+center]

                    val +=  weight* image[x,y]
                    weight_sum += weight

            output_image[i,j] = val/weight_sum if weight_sum != 0 else center_val

    output_image = np.clip(output_image, 0, 255).astype(image.dtype)

    return output_image

def BilateralFilter_SH(image: np.ndarray, r, sigma) -> np.ndarray:
    h, w = image.shape
    output = np.zeros((h, w), dtype=np.uint8)

    # replicate padding
    pad_img = np.zeros((h + 2*r, w + 2*r), dtype=np.float32)

    pad_img[r:r+h, r:r+w] = image
    pad_img[:r, r:r + w] = image[0:1, :]
    pad_img[r + h:, r:r + w] = image[-1:, :]
    pad_img[:, :r] = pad_img[:, r:r + 1]
    pad_img[:, r + w:] = pad_img[:, r + w - 1:r + w]

    # make kernel
    kernel_size = 2 * r + 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - r
            y = j - r
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))

    # Filtering
    for y in range(h):
        for x in range(w):
            center_val = pad_img[y + r, x + r]  # 중심점
            wp_total = 0.0
            filtered_val = 0.0

            for dy in range(-r, r + 1): # 중심점에서 r 내부의 점들에 대해 계산
                for dx in range(-r, r + 1):
                    ny = y + r + dy
                    nx = x + r + dx
                    neighbor_val = pad_img[ny, nx]

                    range_weight = math.exp(-((neighbor_val - center_val) ** 2) / (2 * sigma ** 2)) # 색상 기반 차이
                    weight = kernel[dy + r, dx + r] * range_weight

                    filtered_val += neighbor_val * weight
                    wp_total += weight

            output[y, x] = np.clip(filtered_val / wp_total, 0, 255).astype(np.uint8)

    return output

if __name__ == "__main__":
    main()
