import utilities
import cv2
import os

DATABASE_DIR = "database/optimal/"
RESULTS_DIR = "results/"

def process_single(img1, img2, algo_obj, algo="bfm", only_show=True, return_matches=False,return_kps=False, **params):
    results = None
    if algo == "bfm":
        if only_show:
            utilities.brute_force_orb(img1, img2, orb_obj=algo_obj, draw=True, return_kps=return_kps)
        else:
            results = utilities.brute_force_orb(img1, img2, orb_obj=algo_obj, return_kps=return_kps)
    elif algo == "flann":
        if only_show:
            utilities.flann_executor(img1, img2, algo_obj=algo_obj, draw=True,return_kps=return_kps, **params)
        else:
            results = utilities.flann_executor(img1, img2, algo_obj=algo_obj,return_kps=return_kps, **params)
    else:
        raise ValueError("algo parameter only accepts either 'orb' or 'flann'")
    if not only_show:
        dist_sum = 0
        for match in results[0]:
            dist_sum += match.distance
        if return_matches or return_kps:
            return dist_sum, results
        return dist_sum

def identify(input_img, algo="bfm", return_matches=False, return_kps=False, **params):
    min_dist = None
    best_res = None
    algo_obj = utilities.return_orb_obj(**params)
    for file_name in os.listdir(DATABASE_DIR):
        if file_name[-3:]  == "jpg":
            img2 = cv2.imread(DATABASE_DIR + file_name)
            if return_matches or return_kps:
                dist_sum, results = process_single(input_img, img2, algo_obj=algo_obj, algo=algo,
                                                   only_show=False, return_matches=return_matches,return_kps=return_kps, **params)
            else:
                dist_sum = process_single(input_img, img2, algo_obj=algo_obj, algo=algo, only_show=False, return_kps=return_kps, **params)
            if not min_dist or dist_sum < min_dist:
                min_dist = dist_sum
                match_file = file_name
                if return_matches or return_kps:
                    best_res = results
    if return_matches or return_kps:
        return match_file, best_res
    return match_file


class Processor:
    def __init__(self, files: list, orb_params:dict=None, flann_params:dict=None):
        self.files = files
        self.bfm_results = []
        self.bfm_matches = []
        self.bfm_kps=[]
        self.flann_results = []
        self.flann_matches = []
        self.flann_kps = []
        self.orb_params = orb_params
        self.flann_params = flann_params

    def process_bfm(self, return_matches=False, return_kps=False):
        for file in self.files:
            input_img = cv2.imread(file)
            result = identify(input_img, algo="bfm",return_matches=return_matches,
                              return_kps=return_kps, **self.orb_params)
            if return_matches or return_kps:
                self.bfm_results.append(result[0])
                self.bfm_matches.append(result[1])
            else:
                self.bfm_results.append(result)
            if return_kps:
                self.bfm_kps.append((result[3], result[4]))

    def process_flann(self, return_matches=False, return_kps=False):
        for file in self.files:
            input_img = cv2.imread(file)
            result = identify(input_img, algo="flann", return_matches=return_matches,
                              return_kps=return_kps, **self.flann_params)
            if return_matches or return_kps:
                self.flann_results.append(result[0])
                self.flann_matches.append(result[1])
            else:
                self.flann_results.append(result)
            if return_kps:
                self.flann_kps.append((result[3], result[4]))

    def save_matches(self):
        if self.bfm_matches:
            for file, matches, kps, match_file in zip(self.files, self.bfm_matches, self.bfm_kps, self.bfm_results):
                img1 = cv2.imread(file)
                img2 = cv2.imread(match_file)
                img3 = cv2.drawMatches(img1, kps[0], img2, kps[1], matches,
                                       outImg=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imwrite(RESULTS_DIR+file[:-4]+"_match.png", img3)
        if self.flann_matches:
            for file, matches, kps, match_file in zip(self.files, self.flann_matches, self.flann_kps, self.flann_results):
                img1 = cv2.imread(file)
                img2 = cv2.imread(match_file)
                img3 = cv2.drawMatches(img1, kps[0], img2, kps[1], matches,
                                       outImg=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imwrite(RESULTS_DIR+file[:-4]+"_match.png", img3)

