import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def plot(subplot, image, title):
    plt.subplot(subplot)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis(False)

def gamma_transform(grayscale_image, c, gamma):
    return rescale_image(c * (grayscale_image ** gamma))

def rescale_image(grayscale_image):
    max_val = np.max(grayscale_image)
    res = (grayscale_image / max_val) * 255
    return res

def gamma_mapping(n, c, gamma):
    return c * (n ** gamma)

def rescale_gamma_mapping(cur_val, max_mapped_val, start, end):
    return start + (cur_val / max_mapped_val) * (end - start)

def range_gamma_transform(grayscale_image, start, end, c, gamma):
    mask = (grayscale_image >= start) & (grayscale_image <= end)
    max_val = np.max(grayscale_image[mask])
    max_mapped_val = gamma_mapping(max_val, c, gamma)

    return np.where(
        mask,
        rescale_gamma_mapping(
            gamma_mapping(grayscale_image, c, gamma),
            max_mapped_val, start, end
        ),
        grayscale_image
    )


image = cv.imread('images/Buildings.jpg', cv.IMREAD_GRAYSCALE)

gt_image = gamma_transform(image, 1, 0.3)
rgt_image = range_gamma_transform(image, 0, 100, 1, 0.2)
composed_rgt_image = range_gamma_transform(rgt_image, 120, 255, 1, 5)

cv.imwrite('images/OutputBuildings.jpg', composed_rgt_image)

plot(221, image, 'Orginal Image')
plot(222, gt_image, 'Gamma Transformation with y = 0.3')
plot(223, rgt_image, 'Intensity Range [0, 100] Transformed with y = 0.2')
plot(224, composed_rgt_image, '[0, 100] with y = 0.2 and [120, 255] with y = 5')

plt.show()