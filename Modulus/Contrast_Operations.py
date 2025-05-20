import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import tempfile
import cv2


def manuel_contrast_streching(image, in_min, in_max):
    out_min=0
    out_max=255
    stretched_image = np.clip((image - in_min) / (in_max - in_min) * (out_max - out_min) + out_min, out_min, out_max)
    return stretched_image.astype(np.uint8)


def contrast(image, contrast_slider):
    f = 131 * (contrast_slider + 127) / (127 * (131 - contrast_slider))
    alpha_c = f
    gamma_c = 127 * (1 - f)
    contrasted = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
    return contrasted


def brightness(image, brightness_slider):
    bright = cv2.convertScaleAbs(image, alpha=1, beta=brightness_slider)
    return bright

def histogram(image):
    if image is None:
        return None

    temp_path = os.path.join(tempfile.gettempdir(), "histogram.png")

    if len(image.shape) == 2 or image.shape[2] == 1:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.figure()
        plt.title("Histogram")
        plt.xlabel("Piksel Değeri")
        plt.ylabel("Frekans")
        plt.plot(hist, color='k')
        plt.xlim([0, 256])
        plt.savefig(temp_path)
        plt.close()
    else:
        colors = ('b', 'g', 'r')
        plt.figure()
        plt.title("Histogram")
        plt.xlabel("Piksel Değeri")
        plt.ylabel("Frekans")
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
        plt.xlim([0, 256])
        plt.savefig(temp_path)
        plt.close()

    return cv2.imread(temp_path)

def histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)

    hist_original = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_equalized = cv2.calcHist([equalized], [0], None, [256], [0, 256])

    temp_path = os.path.join(tempfile.gettempdir(), "hist_eq.png")
    plt.figure(figsize=(8,4))
    plt.subplot(1, 2, 1)
    plt.plot(hist_original, color='r')
    plt.title("Orijinal Histogram")
    plt.subplot(1, 2, 2)
    plt.plot(hist_equalized, color='g')
    plt.title("Eşitlenmiş Histogram")
    plt.tight_layout()
    plt.savefig(temp_path)
    plt.close()

    hist_img = cv2.imread(temp_path)
    processed_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    return processed_image, hist_img


def thresholding(image, threshold_value):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x, y = gray.shape
    result = np.zeros_like(gray)

    for i in range(x):
        for j in range(y):
            result[i, j] = 255 if gray[i, j] >= threshold_value else 0

    return result
