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

    # filtered_image_cv = GuidedFilter_SJ(image, image, kernel_size, epsilon)

    noisy = image + np.random.normal(0, 10, image.shape)

    filtered_image_cv = GuidedFilter_SJ(image, noisy, radius=8, epsilon=100)

    cv2.imshow(f"OpenCV GuidedFilter (k = {kernel_size}, epsilon = {epsilon})", filtered_image_cv)
    cv2.imshow("Origin image", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()   


def summed_area_table(image: np.ndarray):
    h, w = image.shape
    res = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            res[i, j] = image[i, j] 
            if j>0: res[i, j] += res[i, j-1]
            if i>0: res[i, j] += res[i-1, j] 
            if i>0 and j>0: res[i, j] -= res[i-1, j-1]
    return res

def mean_filter(image: np.ndarray, r: int):
    h, w = image.shape
    summed = summed_area_table(image)
    res = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            x1, y1 = max(i-r,0), max(j-r,0)
            x2, y2 = min(i+r,h-1), min(j+r,w-1)

            left_top = summed[x1-1, y1-1] if x1>0 and y1>0 else 0
            left_bottom = summed[x1-1, y2] if x1 >0 else 0
            right_top = summed[x2, y1-1] if y1 >0 else 0
            right_bottom = summed[x2, y2]
        
            kernal = (x2-x1+1)*(y2-y1+1)
            res[i, j]= right_bottom - left_bottom - right_top + left_top
            res[i,j] /=kernal

    return res


def GuidedFilter_SJ(guide_image: np.ndarray, image: np.ndarray, radius: int, epsilon: float) -> np.ndarray:
    I = guide_image
    p = image
    r = radius

    mean_I = mean_filter(I, r)
    mean_p = mean_filter(p, r)
    corr_I = mean_filter(I*I, r)
    corr_Ip = mean_filter(I*p, r)

    var_I = corr_I - mean_I*mean_I
    cov_Ip = corr_Ip - mean_I*mean_p

    a = cov_Ip / (var_I + epsilon)
    b = mean_p - a*mean_I

    mean_a = mean_filter(a, r)
    mean_b = mean_filter(b, r)

    output_image = mean_a*I + mean_b

    output_image = np.clip(output_image, 0, 255).astype(image.dtype)

    return output_image

def BilateralFilter_SH(image: np.ndarray, r, sigma) -> np.ndarray:
    
    output_image = np.clip(image, 0, 255).astype(image.dtype)

    return output_image

def BilateralFilter_JY(image: np.ndarray, kernel_size: int, sigma: float, sigma_range: float) -> np.ndarray:
    
    output_image = np.clip(image, 0, 255).astype(image.dtype)

    return output_image

if __name__ == "__main__":
    main()
