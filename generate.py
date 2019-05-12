import utilities
import processor
import cv2
import os


def processFolder(directoryname):
    
    for filename in os.listdir(directoryname):
        if filename.endswith('.jpg'):
            print('processando arquivo: ' + filename)
            img = cv2.imread(directoryname + filename, cv2.IMREAD_GRAYSCALE)
            
            #preprocessing and saving
            img = utilities.clahe(img)
            img = utilities.denoising(img)
            img_final = utilities.adaptive_thresholding(img)
            cv2.imwrite('./database/clahe_denoise_adaptive/'+filename, img_final)
    
    os._exit(0)



if __name__ == "__main__":
    
    currency_name = '20_back'
    directoryname = './bill_scans/'
    
    processFolder(directoryname)
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
    #cv2.imwrite('./database/'+currency_name+'.png', img_final)
    cv2.destroyAllWindows()