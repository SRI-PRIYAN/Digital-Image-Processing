import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def plot(subplot, image, title):
    plt.subplot(subplot)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis(False)

def average_filter(grayscale_image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    return cv.filter2D(grayscale_image, -1, kernel)

def laplacian_filter(grayscale_image):
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    return cv.filter2D(grayscale_image, cv.CV_64F, kernel)

def rescale_laplacian(grayscale_image):
    grayscale_image = grayscale_image - np.min(grayscale_image)
    return ((grayscale_image / np.max(grayscale_image)) * 255).astype(np.uint8)

def smoothen():
    image = cv.imread('images/NoisyImage.png', cv.IMREAD_GRAYSCALE)

    avg_image_7x7 = average_filter(image, 7)
    gaussian_blur_11x11 = cv.GaussianBlur(image, (11, 11), 0)
    gaussian_blur_17x17 = cv.GaussianBlur(image, (17, 17), 0)

    plot(221, image, 'Orginal Image')
    plot(222, avg_image_7x7, 'Averaging with 7x7 kernel')
    plot(223, gaussian_blur_11x11, 'Averaging with 11x11 Gaussian kernel')
    plot(224, gaussian_blur_17x17, 'Averaging with 17x17 Gaussian kernel')
    plt.show()


def sharpen():
    image = cv.imread('images/NoisyImage.png', cv.IMREAD_GRAYSCALE)
    image = cv.GaussianBlur(image, (17, 17), 0)

    laplacian = laplacian_filter(image)
    scaled_laplacian = rescale_laplacian(laplacian)
    sharpened_image = cv.addWeighted(image, 1.5, scaled_laplacian, -0.5, 0)
    
    plot(221, image, 'Original Image')
    plot(222, np.clip(laplacian, 0, 255), 'Laplacian Image')
    plot(223, scaled_laplacian, 'Scaled Laplacian Image')
    plot(224, sharpened_image, 'Sharpened Image')
    plt.show()

# smoothen()
# sharpen()