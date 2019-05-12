import utilities
import preprocessing
import cv2

if __name__ == "__main__":
    
    currency_name = '20_back'
    
    #    CARREGA IMAGEM
    img = cv2.imread('bill_scans/'+ currency_name +'.jpg', cv2.IMREAD_GRAYSCALE)
    
    img = cv2.resize(img, (640,400))
    
    
    img_final = utilities.adaptive_thresholding(img)
    
    # MOSTRA IMAGEM PROCESSADA
    cv2.imshow('image', img_final)

    # MOSTRA IMAGEM ORIGINALl
    cv2.imshow('imagem', img)
    cv2.waitKey(0)

    # SALVA IMAGEM
    cv2.imwrite('./database/'+currency_name+'.png', img_final)
    cv2.destroyAllWindows()