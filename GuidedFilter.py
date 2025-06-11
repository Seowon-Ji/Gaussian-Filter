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

def BilateralFilter_SJ(image: np.ndarray, width: int, sigmaColor: float, sigmaSpace: float) -> np.ndarray:
   
    output_image = np.clip(image, 0, 255).astype(image.dtype)

    return output_image

def BilateralFilter_SH(image: np.ndarray, r, sigma) -> np.ndarray:
    
    output_image = np.clip(image, 0, 255).astype(image.dtype)

    return output_image

def BilateralFilter_JY(image: np.ndarray, kernel_size: int, sigma: float, sigma_range: float) -> np.ndarray:
    
    output_image = np.clip(image, 0, 255).astype(image.dtype)

    return output_image

if __name__ == "__main__":
    main()
