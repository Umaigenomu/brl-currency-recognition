import cv2
import imutils


def resize_img(img, height):
    ratio = img.shape[0] / height
    width = img.shape[1] / ratio
    return imutils.resize(img, width=width, height=height)


# Returns a quadrangular contour or None if none are found
def canny_edge_quad_contour_detection(img, resize=False, resize_height=300, contour_precision=0.015):
    if resize:
        img = resize_img(img, height=resize_height)
    # grayscale -> a bit of blur -> canny edge detection
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_filtered = cv2.bilateralFilter(grayed, 11, 17, 17)
    edged = cv2.Canny(gray_filtered, 30, 200)

    # Find contours (the function is non-destructive since ver. 3.2)
    contours = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    actual_contours = imutils.grab_contours(contours)
    sorted_contours = sorted(actual_contours, key=cv2.contourArea, reverse=True)[:10]
    currency_bill_contour = None
    for cnt in sorted_contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, contour_precision * perimeter, True)
        # If the approximated contour has 4 curves, then it's quandragular, and probably represents the bill
        if len(approx) == 4:
            currency_bill_contour = approx
            break
    return currency_bill_contour


def draw_contour(img, contour, show=True, **params):
    if not params:
        params = [-1, (0, 255, 0), 3]
    # In-place operation
    cv2.drawContours(img, [contour], **params)

    if show:
        cv2.imshow("Contours", img)
        cv2.waitKey(0)


