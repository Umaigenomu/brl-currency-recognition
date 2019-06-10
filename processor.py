import utilities
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

DATABASE_DIR = "database/optimal/"
RESULTS_DIR = "./results/"


def preprocess_img_addr(file):
    img1 = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img1 = utilities.clahe(img1)
    img1 = utilities.denoising(img1)
    img1 = utilities.adaptive_thresholding(img1)
    return img1


def preprocess_img(img, clahe=True, denoising=True, thresholding=True):
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if clahe:
        img1 = utilities.clahe(img1)
    if denoising:
        img1 = utilities.denoising(img1)
    if thresholding:
        img1 = utilities.adaptive_thresholding(img1)
    return img1


def process_single(img1, img2, algo_obj=None, algo="bfm", crosscheck=True, return_best=False,
                   only_show=True, return_matches=False, return_kps=False, **params):
    results = None
    if algo == "bfm":
        if only_show:
            utilities.brute_force_orb(img1, img2, crosscheck=crosscheck,
                                      orb_obj=algo_obj, draw=True, return_kps=return_kps, **params)
        else:
            results = utilities.brute_force_orb(img1, img2, crosscheck=crosscheck,
                                                orb_obj=algo_obj, return_kps=return_kps, **params)
    elif algo == "flann":
        if only_show:
            utilities.flann_executor(img1, img2, algo_obj=algo_obj, draw=True, return_kps=return_kps, **params)
        else:
            results = utilities.flann_executor(img1, img2, algo_obj=algo_obj, return_kps=return_kps, **params)
    else:
        raise ValueError("algo parameter only accepts either 'bfm' or 'flann'")
    if not only_show:
        dist_sum = 0
        for match in results[0]:
            if isinstance(match, list):
                if not match or len(match) < 2:
                    continue
                dist = match[0].distance
            else:
                dist = match.distance
            dist_sum += dist
        if return_matches or return_kps:
            if return_kps:
                if isinstance(results[0][0], list):
                    results = results[:][0]
            else:
                if return_best:
                    best = []
                    for match in results[0]:
                        if not match or len(match) < 2:
                            continue
                        if match[0].distance < 0.75 * match[1].distance:
                            best.append(match[0])
                    results = best
                else:
                    results = results[0]
            return dist_sum, results
        return dist_sum


def identify(input_img, algo="bfm", return_matches=False, return_kps=False, **params):
    min_dist = None
    best_res = None
    algo_obj = utilities.return_orb_obj(**params)
    for file_name in os.listdir(DATABASE_DIR):
        if file_name[-3:] == "jpg":
            img2 = cv2.imread(DATABASE_DIR + file_name)
            if return_matches or return_kps:
                dist_sum, results = process_single(input_img, img2, algo_obj=algo_obj, algo=algo,
                                                   only_show=False, return_matches=return_matches,
                                                   return_kps=return_kps, **params)
            else:
                dist_sum = process_single(input_img, img2, algo_obj=algo_obj, algo=algo, only_show=False,
                                          return_kps=return_kps, **params)
            if not min_dist or dist_sum < min_dist:
                min_dist = dist_sum
                match_file = DATABASE_DIR + file_name
                if return_matches or return_kps:
                    best_res = results
    if return_matches or return_kps:
        final_res = [match_file, *best_res]
        return final_res
    return match_file


class Processor:
    def __init__(self, files: list, orb_params: dict = None, flann_params: dict = None):
        self.files = files
        self.bfm_results = []
        self.bfm_matches = []
        self.bfm_kps = []
        self.flann_results = []
        self.flann_matches = []
        self.flann_kps = []
        if not orb_params:
            self.orb_params = dict()
        else:
            self.orb_params = orb_params
        if not flann_params:
            self.flann_params = dict()
        else:
            self.flann_params = flann_params

    def process_bfm(self, save_matches=False, save_kps=False):
        for file in self.files:
            input_img = preprocess_img_addr(file)
            result = identify(input_img, algo="bfm", return_matches=save_matches,
                              return_kps=save_kps, **self.orb_params)
            if save_matches or save_kps:
                self.bfm_results.append(result[0])
                self.bfm_matches.append(result[1])
            else:
                self.bfm_results.append(result)
            if save_kps:
                self.bfm_kps.append((result[2], result[3]))

    def process_flann(self, save_matches=False, save_kps=False):
        for file in self.files:
            input_img = preprocess_img_addr(file)
            result = identify(input_img, algo="flann", return_matches=save_matches,
                              return_kps=save_kps, **self.flann_params)
            if save_matches or save_kps:
                self.flann_results.append(result[0])
                self.flann_matches.append(result[1])
            else:
                self.flann_results.append(result)
            if save_kps:
                self.flann_kps.append((result[2], result[3]))

    def print_matches(self, save=True, show=False, return_matched_files=True):
        matched_files = []
        if self.bfm_matches:
            for file, matched_files, kps, match_file in zip(self.files, self.bfm_matches, self.bfm_kps,
                                                            self.bfm_results):
                img1 = preprocess_img_addr(file)
                img2 = cv2.imread(match_file)
                print(file)
                print(match_file)
                img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3),
                                       dtype=np.uint8)
                good_matches = sorted(matched_files, key=lambda match: match.distance)[:15]
                img3 = cv2.drawMatches(img1, kps[0], img2, kps[1], good_matches,
                                       outImg=img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                if save:
                    savefile = file.split('/')[1][:-4]
                    cv2.imwrite(RESULTS_DIR + savefile + "_match.png", img3)
                    cv2.imwrite(RESULTS_DIR + savefile + "_match.png", img_matches)
                if show:
                    plt.imshow(img_matches)
                    plt.show()
                if return_matched_files:
                    matched_files.append(match_file)
        elif self.flann_matches:
            for file, matched_files, kps, match_file in zip(self.files, self.flann_matches, self.flann_kps,
                                                            self.flann_results):
                img1 = preprocess_img_addr(file)
                img2 = cv2.imread(match_file)
                print(file)
                print(match_file)
                img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3),
                                       dtype=np.uint8)
                img3 = cv2.drawMatches(img1, kps[0], img2, kps[1], matched_files,
                                       outImg=img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                if save:
                    cv2.imwrite(RESULTS_DIR + file[:-4] + "_match.png", img3)
                    cv2.imwrite(RESULTS_DIR + file[:-4] + "_match.png", img_matches)
                if show:
                    plt.imshow(img_matches)
                    plt.show()
                if return_matched_files:
                    matched_files.append(match_file)

        return matched_files
