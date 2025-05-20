import cv2
import numpy as np

def perspective_correction(image):
    pts1 = np.float32([[100, 200], [500, 150], [120, 600], [520, 650]])
    pts2 = np.float32([[0, 0], [400, 0], [0, 500], [400, 500]])
    
    def angle_between_points(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(cos_theta) * 180 / np.pi
        return angle
    
    def check_perspective(pts):
        pts = pts.reshape(4, 2)
        angles = []
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i+1) % 4]
            p3 = pts[(i+2) % 4]
            angle = angle_between_points(p1, p2, p3)
            angles.append(angle)
        for a in angles:
            if not (70 < a < 110):
                return True  
        return False  

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if check_perspective(pts1):
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        width, height = 400, 500
        warped_image = cv2.warpPerspective(img_rgb, matrix, (width, height))
        return warped_image
    else:
        return img_rgb

def interactive_perspective_correction(image, output_size=(500, 500)):
    selected_points = []
    img_copy = image.copy()

    def select_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(selected_points) < 4:
                selected_points.append((x, y))

                cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Görüntü - Noktaları Seç", img_copy)

            if len(selected_points) == 4:
                cv2.destroyAllWindows()

    cv2.imshow("Görüntü - Noktaları Seç", img_copy)
    cv2.setMouseCallback("Görüntü - Noktaları Seç", select_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(selected_points) != 4:
        print("4 nokta seçilmedi, işlem iptal edildi.")
        return image  

    pts1 = np.float32(selected_points)
    width, height = output_size
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(image, matrix, (width, height))

    return warped