import cv2
import numpy as np

def separate_RGB_channels(image):
    b, g, r = cv2.split(image)

    blue_image = cv2.merge([b, np.zeros_like(g), np.zeros_like(r)])
    green_image = cv2.merge([np.zeros_like(b), g, np.zeros_like(r)])
    red_image = cv2.merge([np.zeros_like(b), np.zeros_like(g), r])

    return red_image, green_image, blue_image

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def negative(image):
    if image is not None:
        return 255 - image
    return None

def gray_negative(image):
    if image is not None:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return 255 - gray_img
    return None

def half_negative(image):
    if image is None:
        return None

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  

    x, y, z = image.shape
    half_negative_image = np.zeros((x, y, z), dtype=np.uint8)

    for i in range(x):
        for j in range(y):
            for k in range(z):
                if i < (x // 2):
                    half_negative_image[i, j, k] = 255 - image[i, j, k]
                else:
                    half_negative_image[i, j, k] = image[i, j, k]
    return half_negative_image