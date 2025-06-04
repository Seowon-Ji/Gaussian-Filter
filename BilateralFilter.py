import numpy as np
import cv2

def main():
    image = cv2.imread("image/cameraman.png", cv2.IMREAD_GRAYSCALE)

    cv2.imshow("Original Image", image)

    kernel_size = 5
    sigma = 10.0
    sigma_range = 30.0

    filtered_image_JY = BilateralFilter_JY(image, kernel_size, sigma, sigma_range)
    cv2.imshow("Filtered JY", filtered_image_JY)
    filtered_image_cv = cv2.bilateralFilter(image, kernel_size, sigma_range, sigma, borderType=cv2.BORDER_REPLICATE)
    cv2.imshow("Filtered CV", filtered_image_cv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

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