"""
Created on 10 de mai de 2019

identifica cedulas e avalia se eh uma nota verdadeira
"""

import cv2
import processor
import utilities
import colors_contours_and_alignment as cca

files_to_check = [
    "bill_shots/2_back_shot_3.jpg",
    "bill_shots/2_back_shot_backlight.jpg",
    "bill_shots/2_dark_back_shot.jpg",
    "bill_shots/2_dark_front_shot.jpg",
    "bill_shots/2_front_shot_1.jpg",
    "bill_shots/5_back_1.jpg",
    "bill_shots/5_dark_back_shot.jpg",
    "bill_shots/5_front_shot_backlight.jpg",
    "bill_shots/10_front_shot_backlight.jpg",
    "fake/100_reais.jpg",
    "fake/100_reais2.jpg",
    "fake/100_reais3.jpg",
]

# width percentages for extracting the watermark
watermark_bounds = {
    "2_back": (287 / 573, 435 / 573),
    "2_front": (137 / 573, 299 / 573),
    "5_back": (291 / 573, 436 / 573),
    "5_front": (133 / 573, 294 / 573),
    "10_back": (286 / 573, 434 / 573),
    "10_front": (140 / 573, 295 / 573),
    "20_back": (277 / 573, 424 / 573),
    "20_front": (148 / 573, 293 / 573),
    "50_back": (276 / 573, 422 / 573),
    "50_front": (155 / 573, 290 / 573),
    "100_back": (278 / 573, 420 / 573),
    "100_front": (161 / 573, 287 / 573)
}


def sample_exec():
    currency_name = '2_back_shot_3'

    #    CARREGA IMAGEM
    img = cv2.imread('bill_shots/' + currency_name + '.jpg', cv2.IMREAD_GRAYSCALE)

    # img = cv2.resize(img, (640,400))

    # PREPROCESSING
    img = utilities.clahe(img)
    img = utilities.denoising(img)
    img_final = utilities.adaptive_thresholding(img)

    # MOSTRA IMAGEM PROCESSADA
    cv2.imshow('image', img_final)

    # MOSTRA IMAGEM ORIGINALl
    cv2.imshow('imagem', img)
    cv2.waitKey(0)

    # SALVA IMAGEM
    # cv2.imwrite('./results/'+currency_name+'.png', img_final)
    cv2.destroyAllWindows()


def identify():
    processor_obj = processor.Processor(files=files_to_check)
    # Saving individual matches is necessary for plotting them later
    processor_obj.process_bfm(save_matches=True, save_kps=True)
    # processor_obj.process_flann(return_matches=True, return_kps=True)
    # Salva os features pareados
    processor_obj.print_matches(save=False)


def extract_watermark_from_bill(bill_img, identity):
    left_b, right_b = watermark_bounds[identity]
    return cca.horizontal_crop(bill_img, left_b, right_b)


# Returns a boolean indicating whether the bill is false or not
def detect_counterfeit(orig_img_addr, test_img_addr, identity, draw_cont=False):
    orig_img = cv2.imread(orig_img_addr)
    test_img = cv2.imread(test_img_addr)

    # orig_gray = cv2.cvtColor(orig_img.copy(), cv2.COLOR_BGR2GRAY)
    # test_gray = cv2.cvtColor(test_img.copy(), cv2.COLOR_BGR2GRAY)

    actual_bill = cca.extract_bill_from_img(test_img, gray_filter=True, draw=draw_cont)
    if actual_bill is None:
        print("Contour not found through edge detection. Using alignment instead.")
        actual_bill = cca.align(orig_img, test_img)[0]

    orig_preprocessed = processor.preprocess_img(orig_img,
                                                 clahe=1,
                                                 denoising=1,
                                                 thresholding=1)
    test_preprocessed = processor.preprocess_img(actual_bill,
                                                 clahe=1,
                                                 denoising=1,
                                                 thresholding=1)

    orig_watermark = extract_watermark_from_bill(orig_preprocessed, identity)
    test_watermark = extract_watermark_from_bill(test_preprocessed, identity)

    # cv2.imshow("orig", orig_watermark)
    # cv2.waitKey(0)
    # cv2.imshow("test", test_watermark)
    # cv2.waitKey(0)

    distances_sum = processor.process_single(orig_watermark, test_watermark, algo="bfm", crosscheck=True, only_show=False)
    print(distances_sum)

    # orig_wm_hsv = cv2.cvtColor(
    #     cv2.cvtColor(orig_watermark, cv2.COLOR_GRAY2BGR),
    #     cv2.COLOR_BGR2HSV)
    # test_hm_hsv = cv2.cvtColor(
    #     cv2.cvtColor(test_watermark, cv2.COLOR_GRAY2BGR),
    #     cv2.COLOR_BGR2HSV)



if __name__ == '__main__':
    detect_counterfeit("bill_scans/100_front.jpg", files_to_check[-2], "100_front")
    detect_counterfeit("bill_scans/100_front.jpg", files_to_check[-3], "100_front")
    detect_counterfeit("bill_scans/100_front.jpg", files_to_check[-1], "100_front")
    print()
    detect_counterfeit("bill_scans/2_back.jpg", files_to_check[0], "2_back")
    detect_counterfeit("bill_scans/2_back.jpg", files_to_check[2], "2_back")
    detect_counterfeit("bill_scans/2_back.jpg", files_to_check[1], "2_back")
