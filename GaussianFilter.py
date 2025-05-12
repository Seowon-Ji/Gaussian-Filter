import cv2 
import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import indices


def main():
    image = cv2.imread("image/cameraman.png", cv2.IMREAD_GRAYSCALE)

    # test code
    cv2.imshow("OpenCV Test", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Gaussian Filter
    GaussianFilter(image)


def GaussianFilter(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:


if __name__ == "__main__":
    main()