"""
Created on 10 de mai de 2019

identifica cedulas e avalia se eh uma nota verdadeira
"""
import cv2
import preprocessing


if __name__ == '__main__':
    
    currency_name = '5_front_shot_backlight'
    #    CARREGA IMAGEM
    img = cv2.imread('bill_shots/'+ currency_name +'.jpg', cv2.IMREAD_GRAYSCALE)
    
    img = cv2.resize(img, (640,400))
    
    
    img_final = preprocessing.process_2back(img)

    # MOSTRA IMAGEM PROCESSADA
    cv2.imshow('image', img_final)

    # MOSTRA IMAGEM ORIGINALl
    cv2.imshow('imagem', img)
    cv2.waitKey(0)

    # SALVA IMAGEM
    #cv2.imwrite('./results/'+currency_name+'.png', img_final)
    cv2.destroyAllWindows()
