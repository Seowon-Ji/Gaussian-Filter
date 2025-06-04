import cv2
import numpy as np
import math

def main():
    image = cv2.imread("image/cameraman.png", cv2.IMREAD_GRAYSCALE) # image read as gray scale

    # Bilateral Filter
    filtered_img = bilateral_filter(image, 10, 3)

    # Print Filtered Image
    cv2.imshow("Original Image", image)
    cv2.imshow("Filtered Image", filtered_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def bilateral_filter(image: np.ndarray, r, sigma) -> np.ndarray:
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