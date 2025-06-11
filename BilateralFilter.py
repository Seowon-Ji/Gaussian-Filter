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

if __name__ == "__main__":
    main()
