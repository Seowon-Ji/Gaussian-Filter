# GuidedFilter
import cv2
import numpy as np

def main():
    # image read as gray scale
    image = cv2.imread("image/cameraman.png", cv2.IMREAD_GRAYSCALE) # (height, width)

    kernel_size = 5
    epsilon = 100.0

    noise = np.random.normal(0, np.random.uniform(5, 25), image.shape).astype(np.float32)
    noise_image = np.clip(image + noise, 0, 255)

    filtered_image_cv = GuidedFilter_SH(image, noise_image, kernel_size, epsilon)

    cv2.imshow("Origin image", image)
    cv2.imshow("Noise image", noise_image.astype(np.float32))
    cv2.imshow(f"OpenCV GuidedFilter (k = {kernel_size}, epsilon = {epsilon})", filtered_image_cv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def summed_area_table(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    sat = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            left = sat[i, j - 1] if j > 0 else 0
            top = sat[i - 1, j] if i > 0 else 0
            corner = sat[i - 1, j - 1] if i > 0 and j > 0 else 0

            sat[i, j] = img[i, j] + left + top - corner

    return sat

def mean_filter(image: np.ndarray, r: int):
    h, w = image.shape
    sat = summed_area_table(image)
    res = np.zeros_like(image, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            x1, y1 = max(i - r, 0), max(j - r, 0)
            x2, y2 = min(i + r, h - 1), min(j + r, w - 1)

            A = sat[x1 - 1, y1 - 1] if x1 > 0 and y1 > 0 else 0
            B = sat[x1 - 1, y2] if x1 > 0 else 0
            C = sat[x2, y1 - 1] if y1 > 0 else 0
            D = sat[x2, y2]

            window_sum = D - B - C + A
            window_area = (x2 - x1 + 1) * (y2 - y1 + 1)

            res[i, j] = window_sum / window_area

    return res

def GuidedFilter_SH(guided_image, noise_image, kernel_size, epsilon):
    p = noise_image.astype(np.float32)
    I = guided_image.astype(np.float32)
    r = kernel_size

    mean_I = mean_filter(I, r)
    mean_p = mean_filter(p, r)
    mean_Ip = mean_filter(I * p, r)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = mean_filter(I * I, r)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / np.maximum(var_I + epsilon, 1e-6)
    b = mean_p - a * mean_I

    mean_a = mean_filter(a, r)
    mean_b = mean_filter(b, r)

    q = mean_a * I + mean_b
    q = np.clip(q, 0, 255).astype(np.uint8)

    return q


if __name__ == "__main__":
    main()