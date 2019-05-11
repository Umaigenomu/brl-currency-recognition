"""
Created on 10 de mai de 2019

identifica cedulas e avalia se eh uma nota verdadeira
"""
import cv2


if __name__ == '__main__':

    img = cv2.imread('bill_scans/2_back.jpg', cv2.IMREAD_GRAYSCALE)
    """
    bin_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,9,5)
    """

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (1, 1), 0)
    ret3, bin_img = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # mostra imagem binarizada
    cv2.imshow('image', bin_img)

    # mostra imagem original
    cv2.imshow('imagem', img)
    cv2.waitKey(0)

    # salva imagem
    cv2.imwrite('gauss-otsu_bin.png', bin_img)
    cv2.destroyAllWindows()
