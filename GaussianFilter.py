import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    image = cv2.imread("image/cameraman.png", cv2.IMREAD_GRAYSCALE) # image read as gray scale

    # test code
    #cv2.imshow("OpenCV Test", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Gaussian Filter
    filtered_img = GaussianFilter(image, 5, 1)

    # Print Filtered Image
    cv2.imshow("Filtered Image", filtered_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def GaussianFilter(image: np.ndarray, r, sigma) -> np.ndarray:
    # make kernel
    kernel_size = 2*r + 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - r
            y = j - r
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    kernel /= np.sum(kernel)

    # replicate padding
    h, w = image.shape
    pad_img = np.zeros((h + 2*r, w + 2*r), dtype=np.float32)

    pad_img[r:r+h, r:r+w] = image
    pad_img[:r, r:r + w] = image[0:1, :]
    pad_img[r + h:, r:r + w] = image[-1:, :]
    pad_img[:, :r] = pad_img[:, r:r + 1]
    pad_img[:, r + w:] = pad_img[:, r + w - 1:r + w]

    # Filtering
    output = np.zeros_like(image, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = pad_img[i:i + kernel_size, j:j + kernel_size]
            output[i, j] = np.sum(region * kernel)

    output = np.clip(output, 0, 255).astype(np.uint8)

    return output

if __name__ == "__main__":
    main()