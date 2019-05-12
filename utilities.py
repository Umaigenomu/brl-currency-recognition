import numpy as np
import cv2
import matplotlib.pyplot as plt

#**************************** BINARIZAZAO ***********************************
def otsu_thresholding(img: np.ndarray, inc_ret=False):
    # Gaussian filtering
    blur = cv2.GaussianBlur(img, (1, 1), 0)
    # Otsu's thresholding
    ret3, bin_img = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if inc_ret:
        return ret3, bin_img
    return bin_img

def adaptive_thresholding(img : np.ndarray):
    
    return  cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY,91,0)
    
#****************************************************************************

#***************************** EQUALIZACAO DE HISTOGRAMA ********************
def clahe(img: np.ndarray):
    
    clahe = cv2.createCLAHE()
    
    return clahe.apply(img) 
#****************************************************************************

#***************************** REDUCAO DE RUIDO *****************************
def bilateral(img: np.ndarray):
        
    return cv2.bilateralFilter(img,5,10,10)

def gaussian_filter(img: np.ndarray):
    pass
#****************************************************************************


#***************************** IDENTIFICACAO DE FEATURES *****************************
def orb(img: np.ndarray, draw=False, nfeatures=500, scoretype=cv2.ORB_HARRIS_SCORE):
    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=nfeatures, scoreType=scoretype)
    # Find keypoints
    kp = orb.detect(img, None)
    # Compute descriptors
    kp, des = orb.compute(img, kp)

    if draw:
        # Draw only the location of each keypoint
        img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
        plt.imshow(img2)
        plt.show()

    return kp, des

def sift(img: np.ndarray):
    pass
#****************************************************************************


#***************************** FEATURE MATCHING *****************************
def brute_force_orb(img1: np.ndarray, img2: np.ndarray, crosscheck=False, draw=False, **orb_params):
    """
    Applies brute force matching using ORB descriptors. As a consequence,
    the distance between each feature is calculated with cv2.NORM_HAMMING.
    :param img1: Image to be checked upon
    :param img2: Bill scan that represents an original image
    :param crosscheck:
        When cross checking is enabled, the matcher will only return matches in which:
        given a feature (i,j) in img1, (i2,j2) from img2 is the best match from
        img1's perspective, and (i, j) is also the best match from img2's perspective.
        That is, the matching algorithm returns mutual results from both sides.
    :param orb_params: Parameters for the ORB function that returns the descriptors
    :return: A DMatch object. A DMatch object has the following attributes:
         DMatch.distance - Distance between descriptors. The lower, the better it is.
         DMatch.trainIdx - Index of the descriptor in train descriptors
         DMatch.queryIdx - Index of the descriptor in query descriptors
         DMatch.imgIdx - Index of the train image.
    """
    # Find keypoints and descriptors for each image
    kp1, des1 = orb(img1, **orb_params)
    kp2, des2 = orb(img2, **orb_params)
    # Create brute force matcher with NORM_HAMMING as its distance alg
    bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crosscheck)
    matches = bfm.match(des1, des2)
    if draw:
        sorted_matches = sorted(matches, key= lambda match: match.distance)
        img3 = cv2.drawMatches()
    return matches

#****************************************************************************



