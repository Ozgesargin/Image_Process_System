import numpy as np
import cv2
import matplotlib.pyplot as plt

def mean_filter(image, kernel_size=(5, 5)):
    return cv2.blur(image, kernel_size)

def median_filter(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

def gauss_filter(image, kernel_size, sigma):
    return cv2.GaussianBlur(image, kernel_size, sigma)

def conservative_filter(image):
    print("Conservative filter başladı...")  

    filtered_image = image.copy()
    shape = image.shape
    
    if len(shape) == 2:  
        rows, cols = shape
        channels = 1
    else:  
        rows, cols, channels = shape

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if channels == 1:  
                region = image[i-1:i+2, j-1:j+2]
                min_val = np.min(region)
                max_val = np.max(region)
                if image[i, j] < min_val:
                    filtered_image[i, j] = min_val
                elif image[i, j] > max_val:
                    filtered_image[i, j] = max_val
            else:  
                for c in range(channels):
                    region = image[i-1:i+2, j-1:j+2, c]
                    min_val = np.min(region)
                    max_val = np.max(region)
                    if image[i, j, c] < min_val:
                        filtered_image[i, j, c] = min_val
                    elif image[i, j, c] > max_val:
                        filtered_image[i, j, c] = max_val

    print("Conservative filter bitti.")  
    return filtered_image

def crimmins_speckle(image):
    filtered_image = image.copy()
    shape = image.shape

    if len(shape) == 2: 
        rows, cols = shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                center_pixel = image[i, j]
                neighbors = [image[i-1, j], image[i+1, j], image[i, j-1], image[i, j+1]]
                avg_neighbors = np.mean(neighbors)

                if center_pixel > avg_neighbors + 20:
                    filtered_image[i, j] = avg_neighbors
                elif center_pixel < avg_neighbors - 20:
                    filtered_image[i, j] = avg_neighbors

    else:  
        rows, cols, channels = shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                for c in range(channels):
                    center_pixel = image[i, j, c]
                    neighbors = [
                        image[i-1, j, c],
                        image[i+1, j, c],
                        image[i, j-1, c],
                        image[i, j+1, c]
                    ]
                    avg_neighbors = np.mean(neighbors)

                    if center_pixel > avg_neighbors + 20:
                        filtered_image[i, j, c] = avg_neighbors
                    elif center_pixel < avg_neighbors - 20:
                        filtered_image[i, j, c] = avg_neighbors

    return filtered_image

def apply_low_pass_filter(image, radius=30):
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)

    rows, cols = image.shape
    mask = np.zeros((rows, cols), np.uint8)
    center = (cols // 2, rows // 2)
    cv2.circle(mask, center, radius, 1, -1)

    filtered = f_shift * mask
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered)).real

    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    return filtered_image

def apply_high_pass_filter(image, radius=30):
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)

    rows, cols = image.shape
    mask = np.ones((rows, cols), np.uint8)
    center = (cols // 2, rows // 2)
    cv2.circle(mask, center, radius, 0, -1)

    filtered = f_shift * mask
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered)).real

    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    return filtered_image

def apply_band_filter(image, D1=20, D2=50, band_pass=True):
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image

    f_transform = np.fft.fft2(image_gray)
    f_shift = np.fft.fftshift(f_transform)

    rows, cols = image_gray.shape
    center = (cols // 2, rows // 2)

    mask = np.zeros((rows, cols), np.uint8) if band_pass else np.ones((rows, cols), np.uint8)

    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - center[1])**2 + (v - center[0])**2)
            if band_pass:
                if D1 <= D <= D2:
                    mask[u, v] = 1
            else:
                if D1 <= D <= D2:
                    mask[u, v] = 0

    filtered = f_shift * mask
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered)).real

    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    return filtered_image

def apply_butterworth_filter(image, D0=30, n=2, filter_type="lowpass"):

    import cv2
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    rows, cols = image.shape
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)

    mask = np.zeros((rows, cols), np.float32)
    center = (cols // 2, rows // 2)

    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - center[1])**2 + (v - center[0])**2)
            if filter_type == "lowpass":
                H = 1 / (1 + (D / D0)**(2 * n))
            elif filter_type == "highpass":
                H = 1 - (1 / (1 + (D / D0)**(2 * n)))
            else:
                raise ValueError("filter_type 'lowpass' veya 'highpass' olmalı.")
            mask[u, v] = H

    filtered = f_shift * mask
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered)).real
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    return filtered_image

def apply_gaussian_filter(image, D0=30, filter_type="lowpass"):

    import cv2
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    rows, cols = image.shape
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)

    mask = np.zeros((rows, cols), np.float32)
    center = (cols // 2, rows // 2)

    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - center[1])**2 + (v - center[0])**2)
            if filter_type == "lowpass":
                H = np.exp(-(D**2) / (2 * (D0**2)))
            elif filter_type == "highpass":
                H = 1 - np.exp(-(D**2) / (2 * (D0**2)))
            else:
                raise ValueError("filter_type 'lowpass' veya 'highpass' olmalı.")
            mask[u, v] = H

    filtered = f_shift * mask
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered)).real
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    return filtered_image

def homomorphic_filter(image, d0=30, h_l=0.5, h_h=2, c=1):
    import cv2
    import numpy as np
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    log_image = np.log1p(np.float32(gray))

    f_transform = np.fft.fft2(log_image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    rows, cols = gray.shape
    center = (cols//2, rows//2)
    H = np.zeros((rows, cols), np.float32)

    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - center[1])**2 + (v - center[0])**2)
            H[u, v] = (h_h - h_l) * (1 - np.exp(-c * (D**2 / d0**2))) + h_l

    filtered = f_transform_shifted * H
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered)).real

    final_image = np.expm1(filtered_image)
    final_image = np.clip(final_image, 0, 255)

    return np.uint8(final_image)

