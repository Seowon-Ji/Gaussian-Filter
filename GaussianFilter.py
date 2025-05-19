import cv2 
import numpy as np
import matplotlib.pyplot as plt

def main():
    image = cv2.imread("image/cameraman.png", cv2.IMREAD_GRAYSCALE)

    # test code
    # cv2.imshow("OpenCV Test", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Gaussian Filter
    width = 7
    sigma = 2
    output_image = GaussianFilter(image, width, sigma)
    cv2.imshow(f"Filtered Image: filter size = {width}, sigma = {sigma}", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def GaussianFilter(image: np.ndarray, width: int, sigma: float) -> np.ndarray:
    center = width//2

    # Create filter size array
    kernel = np.zeros((width, width))

    for i in range(width):
        for j in range(width):
            x = i-center
            y= j-center
            kernel[i,j] = x**2 + y**2

    kernel /= 2*sigma**2
    kernel = np.exp(-kernel)
    kernel /= kernel.sum()

    image_row, image_col = image.shape
    output_image = np.zeros((image_row, image_col))
    padding = width//2

    padded_image = np.zeros((image_row + 2*padding, image_col + 2*padding))
    padded_image[padding:padded_image.shape[0]-padding,  padding:padded_image.shape[1]-padding] = image

    for i in range(image_row):
        for j in range(image_col):
            output_image[i,j] = np.sum(kernel * padded_image[i:i+width, j:j+width])

    output_image = np.clip(output_image, 0, 255).astype(image.dtype)

    return output_image

if __name__ == "__main__":
    main()