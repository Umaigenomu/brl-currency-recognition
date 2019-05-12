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
    orb_obj = cv2.ORB_create(nfeatures=nfeatures, scoreType=scoretype)
    # Find keypoints
    kp = orb_obj.detect(img, None)
    # Compute descriptors
    kp, des = orb_obj.compute(img, kp)

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
def brute_force_orb(img1: np.ndarray, img2: np.ndarray, crosscheck=False,
                    orb_obj=None, draw=False, **orb_params):
    """
    Applies brute force matching using ORB descriptors. As a consequence,
    the distance between each feature is calculated with cv2.NORM_HAMMING.
    :param img1: Image to be checked upon. More specifically, a photo of
                 a BRL currency bill taken by the user.
    :param img2: Bill scan that represents the model to be based upon.
    :param crosscheck:
        When cross checking is enabled, the matcher will only return matches in which:
        given a pair of features (i,j) in img1, and (i2,j2) from img2 is the best match
        from img1's perspective, and (i, j) is also the best match from img2's perspective.
        That is, the matching algorithm returns mutual results from both sides.
    :param orb_params: Parameters for the ORB function that returns the descriptors
    :param orb_obj:
        If creating a new orb object every time this function is called turns out to be
        inefficient, you may opt to use this parameter and employ a previously created
        instance. In that case, orb_params will be ignored.
    :param draw: Set this parameter to True if you want to show an image of the results
                 during runtime.
    :return: A DMatch object. A DMatch object has the following attributes:
         DMatch.distance - Distance between descriptors. The lower, the better it is.
         DMatch.trainIdx - Index of the descriptor in train descriptors
         DMatch.queryIdx - Index of the descriptor in query descriptors
         DMatch.imgIdx - Index of the train image.
    """
    # Find keypoints and descriptors for each image
    if not orb_obj:
        kp1, des1 = orb(img1, **orb_params)
        kp2, des2 = orb(img2, **orb_params)
    else:
        kp1, des1 = orb_obj.detectAndCompute(img1, None)
        kp2, des2 = orb_obj.detectAndCompute(img2, None)
    # Create brute force matcher with NORM_HAMMING as its distance alg
    bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crosscheck)
    matches = bfm.match(des1, des2)
    if draw:
        sorted_matches = sorted(matches, key= lambda match: match.distance)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, sorted_matches[:15],
                               None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3)
        plt.show()
    return matches

def flann(kp1, des1, kp2, des2, index_params=None, search_params=None, flann_obj=None,
          match_count=2, return_best=False):
    """

    :param kp1:
    :param des1:
    :param kp2:
    :param des2:
    :param index_params:
    :param search_params:
    :param flann_obj:
    :param match_count:
    :param return_best:
    :return:
    """
    if not flann_obj:
        flann_obj = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann_obj.knnMatch(des1, des2, k=match_count)
    if return_best:
        # Filter matches using Lowe's ratio test
        good_matches = []
        for kp1, kp2 in matches:
            if kp1.distance < 0.7 * kp2.distance:
                good_matches.append(kp1)
        return good_matches
    else:
        return matches

def flann_executor(img1: np.ndarray, img2: np.ndarray, algo="orb", algo_obj=None, draw=False):
    if not algo_obj:
        if algo == "orb":
            pass
        else:
            pass

    pass
#****************************************************************************



