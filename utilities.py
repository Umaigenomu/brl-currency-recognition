import numpy as np
import cv2


def otsu_thresholding(img: np.ndarray, inc_ret=False) -> np.ndarray:
    # Gaussian filtering
    blur = cv2.GaussianBlur(img, (1, 1), 0)
    # Otsu's thresholding
    ret3, bin_img = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if inc_ret:
        return ret3, bin_img
    return bin_img




