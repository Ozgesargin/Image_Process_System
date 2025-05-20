import cv2
import numpy as np

def shifting(image, Right_Shift, Down_Shift):
    M = np.float32([[1, 0, Right_Shift], [0, 1, Down_Shift]])
    moved_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return moved_image

def shearing(image, sh_x=0, sh_y=0):
    h, w = image.shape[:2]

    if sh_x != 0:
        new_w = w + int(abs(sh_x) * h)
        sheared_image = np.zeros((h, new_w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                x2 = int(x + sh_x * y)
                if 0 <= x2 < new_w:
                    sheared_image[y, x2] = image[y, x]
        return sheared_image

    if sh_y != 0:
        new_h = h + int(abs(sh_y) * w)
        sheared_image = np.zeros((new_h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                y2 = int(y + sh_y * x)
                if 0 <= y2 < new_h:
                    sheared_image[y2, x] = image[y, x]
        return sheared_image

    return image


def add_text_to_image(image, text, scale_factor=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(1 / scale_factor, 0.3)  
    thickness = 2
    color = (0, 0, 0)
    cv2.putText(image, text, (10, 25), font, font_scale, color, thickness)
    return image

def scale_image(image, factor, scale_up=True):
    h, w = image.shape[:2]

    if scale_up:
        new_h, new_w = int(h * factor), int(w * factor)
    else:
        new_h, new_w = int(h / factor), int(w / factor)

    bilinear = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    bicubic = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    lanczos = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    bilinear = add_text_to_image(bilinear, "Bilinear", factor)
    bicubic = add_text_to_image(bicubic, "Bicubic", factor)
    lanczos = add_text_to_image(lanczos, "Lanczos", factor)

    combined = np.hstack((bilinear, bicubic, lanczos))

    return combined

def crop_image(image, x_min, x_max, y_min, y_max):
    return image[x_max:x_max+y_max, x_min:x_min+y_min]

def rotate_image_with_interpolation(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    rot_nearest = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rot_bilinear = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated_nearest = cv2.warpAffine(image, rot_nearest, (w, h), flags=cv2.INTER_NEAREST)
    rotated_bilinear = cv2.warpAffine(image, rot_bilinear, (w, h), flags=cv2.INTER_LINEAR)

    return cv2.hconcat([rotated_nearest, rotated_bilinear])

def mirror_image(image, mirror_angle):
    if mirror_angle == 0:
        return cv2.flip(image, 1)  
    elif mirror_angle == 90:
        return cv2.flip(image, 0)  
    elif mirror_angle == 180:
        return cv2.flip(image, -1)  
    else:
        return image

