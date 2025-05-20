import cv2
import numpy as np
    
def sobel(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    sobel_normalized = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))
    
    return sobel_normalized

def prewitt(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1],
                         [ 0,  0,  0],
                         [ 1,  1,  1]], dtype=np.float32)
    
    prewitt_x = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
    prewitt_y = cv2.filter2D(gray, cv2.CV_32F, kernel_y)
    
    prewitt_magnitude = cv2.magnitude(prewitt_x, prewitt_y)
    prewitt_normalized = np.uint8(255 * prewitt_magnitude / np.max(prewitt_magnitude))
    
    return prewitt_normalized

def robertss_cross(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    roberts_x = np.array([[1, 0],
                          [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, 1],
                          [-1, 0]], dtype=np.float32)
    
    roberts_x_result = cv2.filter2D(gray, cv2.CV_32F, roberts_x)
    roberts_y_result = cv2.filter2D(gray, cv2.CV_32F, roberts_y)
    
    roberts_magnitude = cv2.magnitude(roberts_x_result, roberts_y_result)
    roberts_normalized = np.uint8(255 * roberts_magnitude / np.max(roberts_magnitude))
    
    return roberts_normalized

def compass(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    compass_kernels = [
        np.array([[-1, -1, -1], [1, 1, 1], [1, 1, 1]], dtype=np.float32),  # Doğu (E)
        np.array([[1, 1, 1], [1, 1, 1], [-1, -1, -1]], dtype=np.float32),  # Batı (W)
        np.array([[-1, 1, 1], [-1, 1, 1], [-1, 1, 1]], dtype=np.float32),  # Kuzey (N)
        np.array([[1, 1, -1], [1, 1, -1], [1, 1, -1]], dtype=np.float32)   # Güney (S)
    ]

    compass_edges = np.zeros_like(gray, dtype=np.float32)

    for kernel in compass_kernels:
        edge = cv2.filter2D(gray, cv2.CV_32F, kernel)
        compass_edges = np.maximum(compass_edges, edge)

    compass_normalized = np.uint8(255 * compass_edges / np.max(compass_edges))

    return compass_normalized

def canny(image, threshold1=50, threshold2=150):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    edges = cv2.Canny(gray, threshold1, threshold2)
    return edges

def laplacian(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)  

    return laplacian_abs

def gabor(image, ksize=21, sigma=5, theta=np.pi/4, lambd=10, gamma=0.5, psi=0, ktype=cv2.CV_32F):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype)
    filtered = cv2.filter2D(gray, cv2.CV_8UC3, gabor_kernel)

    return filtered

def apply_hough_lines(image, threshold=50, min_line_length=50, max_line_gap=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5,5), 1.5)
    edges = cv2.Canny(blurred, 50, 150)
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        print("Hiç doğru tespit edilemedi.")
    
    return output

def apply_hough_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 30,
                               param1=50, param2=30, minRadius=10, maxRadius=100)
    output = image.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)
    return output

def apply_kmeans_segmentation(image, k=3):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pixel_values = image_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    ret, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    if labels is not None and centers is not None:
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()].reshape(image_rgb.shape)
        return segmented_image
    else:
        print("K-Means başarılı olmadı veya sonuç yok.")
        return None

