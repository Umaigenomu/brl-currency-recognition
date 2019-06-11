import cv2
import imutils
import numpy as np
from skimage.exposure import rescale_intensity

import utilities


def resize_img(img, height):
    # width is automatically calculated
    return imutils.resize(img, height=height)


# Returns a quadrangular contour or None if none are found. Basically, 4 coordinates in a bizarre np.shape
def canny_edge_quad_contour_detection(img, gray_filter=True, resize=False, resize_height=300, contour_precision=0.015):
    if resize:
        img = resize_img(img, height=resize_height)
    # grayscale -> a bit of blur -> canny edge detection
    if gray_filter:
        grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        input_img = cv2.bilateralFilter(grayed, 11, 17, 17)
    else:
        grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        input_img = utilities.clahe(grayed)
        # input_img = utilities.denoising(input_img)
        # input_img = utilities.adaptive_thresholding(input_img)
        # grayed = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        # input_img = cv2.bilateralFilter(grayed, 11, 17, 17)

    edged = cv2.Canny(input_img, 20, 200)
    cv2.imwrite("results/examples/canny_edge_2_darkback.png", edged)
    # cv2.imshow("test", edged)
    # cv2.waitKey(0)

    # Find contours (the function is non-destructive since ver. 3.2. That is, the image isn't affected.)
    contours = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    actual_contours = imutils.grab_contours(contours)
    sorted_contours = sorted(actual_contours, key=cv2.contourArea, reverse=True)[:10]
    currency_bill_contour = np.array([])
    for cnt in sorted_contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, contour_precision * perimeter, True)
        # If the approximated contour has 4 corners, then it's quandragular, and probably represents the bill
        if len(approx) == 4:
            currency_bill_contour = approx
            break
    return currency_bill_contour


def draw_contour(img, contour, show=True, save=True, *params):
    if not params:
        params = [-1, (0, 255, 0), 3]
    # In-place operation
    cv2.drawContours(img, [contour], *params)

    if show:
        cv2.imshow("Contours", img)
        cv2.waitKey(0)
    if save:
        cv2.imwrite("results/examples/Contours.png", img)

def extract_contour_from_img(original_img, contour, ratio=1):
    # Top-left, top-right, bottom-right, and bottom-left corners of the contour in an
    # unknown order! As such, we'll organize them in a new matrix with the above order.
    corners = contour.reshape(4, 2)
    # Initializing output
    rect = np.zeros((4, 2), dtype="float32")

    point_sums = corners.sum(axis=1)
    # Top-left and bottom-right corners
    rect[0] = corners[np.argmin(point_sums)]
    rect[2] = corners[np.argmax(point_sums)]

    point_diffs = np.diff(corners, axis=1)
    # Top-right and bottom-left corners
    rect[1] = corners[np.argmin(point_diffs)]
    rect[3] = corners[np.argmax(point_diffs)]

    # If the image was resized earlier, then we need to convert the points
    # so that they represent corners from the original image
    rect *= ratio
    tl, tr, br, bl = rect

    # Computing the width and height of our new image by comparing distances
    width_b = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_t = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    height_r = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_l = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    new_width = int(max(width_b, width_t))
    new_height = int(max(height_r, height_l))

    # ignore rotation of points
    dest_points = np.array(
        [
            [0, 0],
            [new_width - 1, 0],
            [new_width - 1, new_height - 1],
            [0, new_height - 1]
        ],
        dtype="float32"
    )

    M = cv2.getPerspectiveTransform(rect, dest_points)
    warped_img = cv2.warpPerspective(original_img, M, (new_width, new_height))
    return warped_img


def horizontal_crop(img, left_bound_perc, right_bound_perc):
    h, w = img.shape[:2]
    left_bound = round(w * left_bound_perc)
    right_bound = round(w * right_bound_perc)
    return img[:, left_bound:right_bound]


def rescale_with_intensity_return_gray(img):
    grayed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rescaled = rescale_intensity(grayed_img, out_range=(0, 255))
    return rescaled


def extract_bill_from_img(img, gray_filter=True, resize=True, resize_height=300, draw=False):
    if resize:
        resized_img = resize_img(img.copy(), resize_height)
        ratio = img.shape[0] / resize_height
        cnt = canny_edge_quad_contour_detection(resized_img, gray_filter=gray_filter, contour_precision=0.015)
        if not cnt.any():
            return None
        if draw:
            draw_contour(resized_img, cnt)
        warped_img = extract_contour_from_img(img, cnt, ratio)
    else:
        cnt = canny_edge_quad_contour_detection(img, contour_precision=0.015)
        if not cnt.any():
            return None
        if draw:
            draw_contour(img, cnt)
        warped_img = extract_contour_from_img(img, cnt)

    # cv2.imwrite(processor.RESULTS_DIR + "warped.png", warped_img)
    return warped_img


# This method turned out to be pretty bad lol
# Aligns the second image with the first using warping through a calculated homography
def align(orig_img, img2, max_features=500, good_match_perc=0.15):
    # Convert images to grayscale
    img1_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(max_features)
    kp1, descriptors1 = orb.detectAndCompute(img1_gray, None)
    kp2, descriptors2 = orb.detectAndCompute(img2_gray, None)

    # Match features.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    num_good_matches = int(len(matches) * good_match_perc)
    matches = matches[:num_good_matches]

    # Draw top matches
    # imMatches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
    # cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Find and use homography. Ransac is an outlier detection algorithm that's used here
    # for excluding bad matches.
    homography, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    height, width, channels = orig_img.shape
    img2_reg = cv2.warpPerspective(img2, homography, (width, height))

    return img2_reg, homography


