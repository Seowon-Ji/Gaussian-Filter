import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import indices

def main():
    # image read as gray scale
    image = cv2.imread("image/cameraman.png", cv2.IMREAD_GRAYSCALE) # (height, width)

    # test code
    # cv2.imshow("OpenCV Test", image)

    kernel_size = 8
    epsilon = 100

    noisy = image + np.random.normal(0, 10, image.shape)

    filtered_image_cv = GuidedFilter_JY(image, noisy, radius=8, epsilon=100)

    cv2.imshow(f"OpenCV GuidedFilter (k = {kernel_size}, epsilon = {epsilon})", filtered_image_cv)
    cv2.imshow("Origin image", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def BilateralFilter_SJ(image: np.ndarray, width: int, sigmaColor: float, sigmaSpace: float) -> np.ndarray:
   
    output_image = np.clip(image, 0, 255).astype(image.dtype)

    return output_image

def BilateralFilter_SH(image: np.ndarray, r, sigma) -> np.ndarray:
    
    output_image = np.clip(image, 0, 255).astype(image.dtype)

    return output_image


def GuidedFilter_JY(guidance: np.ndarray, p: np.ndarray, radius: int, epsilon: float) -> np.ndarray:

    height, width = guidance.shape
    I = guidance.astype(np.float64)
    p_in = p.astype(np.float64)

    a = np.zeros_like(I)
    b = np.zeros_like(I)

    window_area = (2 * radius + 1) ** 2

    for r_k in range(height):
        for c_k in range(width):
            sum_I, sum_p, sum_II, sum_Ip = 0.0, 0.0, 0.0, 0.0

            for kr_offset in range(-radius, radius + 1):
                for kc_offset in range(-radius, radius + 1):
                    r_in = r_k + kr_offset
                    c_in = c_k + kc_offset

                    eff_r_in = max(0, min(r_in, height - 1))
                    eff_c_in = max(0, min(c_in, width - 1))

                    I_val = I[eff_r_in, eff_c_in]
                    p_val = p_in[eff_r_in, eff_c_in]

                    sum_I += I_val
                    sum_p += p_val
                    sum_II += I_val * I_val
                    sum_Ip += I_val * p_val

            mean_I = sum_I / window_area
            mean_p = sum_p / window_area
            mean_II = sum_II / window_area
            mean_Ip = sum_Ip / window_area

            var_I = mean_II - mean_I * mean_I  # E[X^2] - (E[X])^2
            cov_Ip = mean_Ip - mean_I * mean_p  # E[XY] - E[X]E[Y]

            ak = cov_Ip / (var_I + epsilon)
            bk = mean_p - ak * mean_I

            a[r_k, c_k] = ak
            b[r_k, c_k] = bk

    mean_a = np.zeros_like(I)
    mean_b = np.zeros_like(I)

    for r_i in range(height):
        for c_i in range(width):
            sum_a, sum_b = 0.0, 0.0
            for kr_offset in range(-radius, radius + 1):
                for kc_offset in range(-radius, radius + 1):
                    r_in = r_i + kr_offset
                    c_in = c_i + kc_offset

                    eff_r_in = max(0, min(r_in, height - 1))
                    eff_c_in = max(0, min(c_in, width - 1))

                    sum_a += a[eff_r_in, eff_c_in]
                    sum_b += b[eff_r_in, eff_c_in]

            mean_a[r_i, c_i] = sum_a / window_area
            mean_b[r_i, c_i] = sum_b / window_area

    q = mean_a * I + mean_b

    output_image = np.clip(q, 0, 255)
    output_image = output_image.astype(guidance.dtype)

    return output_image

if __name__ == "__main__":
    main()
