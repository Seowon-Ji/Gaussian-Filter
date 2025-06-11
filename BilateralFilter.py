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

def BilateralFilter_JY(image: np.ndarray, kernel_size: int, sigma: float, sigma_range: float) -> np.ndarray:
    # 커널 한 변 길이 검증
    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError("Kernel size must be odd.")

    # 인풋 이미지와 동일한 크기의 빈 이미지 생성
    height, width = image.shape
    output_image = np.zeros_like(image, dtype=np.float64)  # 연산 중 소수점 발생 가능

    radius = kernel_size // 2

    indices = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(indices, indices)

    for r_out in range(height):
        for c_out in range(width):

            center_pixel_intensity = image[r_out, c_out].astype(np.float64)

            weighted_sum = 0.0
            total_weight = 0.0

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

                    neighbor_pixel_intensity = image[eff_r_in, eff_c_in].astype(np.float64)

                    spatial_gaussian_weight = np.exp(-(kr_offset ** 2 + kc_offset ** 2) / (2 * sigma ** 2))

                    intensity_difference_sq = (center_pixel_intensity - neighbor_pixel_intensity) ** 2

                    range_gaussian_weight = np.exp(-intensity_difference_sq / (2 * sigma_range ** 2))

                    bilateral_weight = spatial_gaussian_weight * range_gaussian_weight

                    weighted_sum += bilateral_weight * neighbor_pixel_intensity
                    total_weight += bilateral_weight

            if total_weight > 0:
                output_image[r_out, c_out] = weighted_sum / total_weight
            else:
                output_image[r_out, c_out] = center_pixel_intensity

    # 최종 값 범위 조정 및 타입 변환
    output_image = np.clip(output_image, 0, 255)
    output_image = output_image.astype(image.dtype)

    return output_image

if __name__ == "__main__":
    main()
