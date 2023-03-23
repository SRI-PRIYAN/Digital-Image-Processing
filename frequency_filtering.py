import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def plot(subplot, image, title):
    plt.subplot(subplot)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis(False)


def D(u, v, c_row, c_col):
    return np.sqrt((u - c_row)**2 + (v - c_col)**2)


def D2(u, v, c_row, c_col):
    return (u - c_row)**2 + (v - c_col)**2


def ideal_low_pass_filter(shifted_ft_image, size):
    rows, cols = shifted_ft_image.shape
    c_row, c_col = rows // 2, cols // 2

    mask = np.zeros(shifted_ft_image.shape)
    u, v = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    mask[D(u, v, c_row, c_col) <= size] = 1

    return shifted_ft_image * mask


def gaussian_low_pass_filter(shifted_ft_image, size):
    rows, cols = shifted_ft_image.shape
    c_row, c_col = rows // 2, cols // 2

    u, v = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    mask = np.exp((-D2(u, v, c_row, c_col)) / (2 * size**2))

    return shifted_ft_image * mask


def ideal_high_pass_filter(shifted_ft_image, size):
    rows, cols = shifted_ft_image.shape
    c_row, c_col = rows // 2, cols // 2

    mask = np.ones(shifted_ft_image.shape)
    u, v = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    mask[D(u, v, c_row, c_col) <= size] = 0

    return shifted_ft_image * mask


def gaussian_high_pass_filter(shifted_ft_image, size):
    rows, cols = shifted_ft_image.shape
    c_row, c_col = rows // 2, cols // 2
    
    u, v = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    mask = 1 - np.exp((-D2(u, v, c_row, c_col)) / (2 * size**2))

    return shifted_ft_image * mask


def filter_image(image, filter, filter_size):
    # Apply Fourier Transform to the Image
    ft_image = np.fft.fft2(image)
    shifted_ft_image = np.fft.fftshift(ft_image)
    magnitude_spectrum = 20 * np.log(np.abs(shifted_ft_image))

    # Filter the Transformed Image
    filtered_shifted_ft_image = filter(shifted_ft_image, filter_size)
    filtered_magnitude_spectrum = 20 * np.log(
        np.abs(filtered_shifted_ft_image) + 1
    )

    # Inverse Fourier Transform to get the filtered image
    filtered_ft_image = np.fft.ifftshift(filtered_shifted_ft_image)
    filtered_image = np.real(np.fft.ifft2(filtered_ft_image))

    return magnitude_spectrum, filtered_magnitude_spectrum, filtered_image


def smoothen(image, low_pass_filter, filter_size):
    magnitude_spectrum, smoothened_magnitude_spectrum, smoothened_image\
          = filter_image(image, low_pass_filter, filter_size)

    plot(221, image, 'Input Image')
    plot(222, magnitude_spectrum, 'Magnitude Spectrum')
    plot(223, smoothened_magnitude_spectrum, 'Smoothened Magnitude Spectrum')
    plot(224, smoothened_image, 'Smoothened Image')
    plt.show()


def sharpen(image, high_pass_filter, filter_size):
    magnitude_spectrum, sharpened_magnitude_spectrum, sharpened_image\
          = filter_image(image, high_pass_filter, filter_size)

    plot(221, image, 'Input Image')
    plot(222, magnitude_spectrum, 'Magnitude Spectrum')
    plot(223, sharpened_magnitude_spectrum, 'Sharpened Magnitude Spectrum')
    plot(224, sharpened_image, 'Sharpened Image')
    plt.show()


# image = cv.imread('images/RohitSharma.jpg', cv.IMREAD_GRAYSCALE)
image = cv.imread('images/NoisyImage.png', cv.IMREAD_GRAYSCALE)
image = cv.GaussianBlur(image, (17, 17), 0)
# image = cv.imread('images/OutputBuildings.jpg', cv.IMREAD_GRAYSCALE)

# smoothen(image, ideal_low_pass_filter, 60)
# smoothen(image, gaussian_low_pass_filter, 50)
sharpen(image, ideal_high_pass_filter, 20)
sharpen(image, gaussian_high_pass_filter, 20)