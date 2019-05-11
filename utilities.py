import numpy as np
import cv2
import matplotlib.pyplot as plt

def otsu_thresholding(img: np.ndarray, inc_ret=False):
    # Gaussian filtering
    blur = cv2.GaussianBlur(img, (1, 1), 0)
    # Otsu's thresholding
    ret3, bin_img = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if inc_ret:
        return ret3, bin_img
    return bin_img


def obr(img: np.ndarray, draw=False):
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # Find keypoints
    kps = orb.detect(img, None)
    # Compute descriptors
    kps, des = orb.compute(img, kps)

    if draw:
        # Draw only the location of each keypoint
        img2 = cv2.drawKeypoints(img, kps, None, color=(0,255,0), flags=0)
        plt.imshow(img2)
        plt.show()

    return kps, des

