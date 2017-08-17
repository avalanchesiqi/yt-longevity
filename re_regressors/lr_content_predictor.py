#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Predict relative engagement from content, with ridge regression."""

from __future__ import division, print_function
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge

from utils.helper import write_dict_to_pickle


def ridge_regression(train_x, train_y, test_x):
    # ridge regression
    ridge_model = Ridge(fit_intercept=True)
    ridge_model.fit(train_x, train_y)
    test_yhat = ridge_model.predict(test_x)
    return test_yhat


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    category_dict = {'1': 0, '2': 1, '10': 2, '15': 3, '17': 4, '19': 5, '20': 6, '22': 7, '23': 8, '24': 9,
                     '25': 10, '26': 11, '27': 12, '28': 13, '29': 14, '30': 15, '34': 16, '35': 17, '43': 18, '44': 19}
    category_cnt = len(category_dict)
    lang_dict = {'af': 0, 'ar': 1, 'bg': 2, 'bn': 3, 'ca': 4, 'cs': 5, 'cy': 6, 'da': 7, 'de': 8, 'el': 9, 'en': 10,
                 'es': 11, 'et': 12, 'fa': 13, 'fi': 14, 'fr': 15, 'gu': 16, 'he': 17, 'hi': 18, 'hr': 19, 'hu': 20,
                 'id': 21, 'it': 22, 'ja': 23, 'kn': 24, 'ko': 25, 'lt': 26, 'lv': 27, 'mk': 28, 'ml': 29, 'mr': 30,
                 'ne': 31, 'nl': 32, 'no': 33, 'pa': 34, 'pl': 35, 'pt': 36, 'ro': 37, 'ru': 38, 'sk': 39, 'sl': 40,
                 'so': 41, 'sq': 42, 'sv': 43, 'sw': 44, 'ta': 45, 'te': 46, 'th': 47, 'tl': 48, 'tr': 49, 'uk': 50,
                 'ur': 51, 'vi': 52, 'zh-cn': 53, 'zh-tw': 54, 'NA': 55}
    lang_cnt = len(lang_dict)

    true_value = []
    predict_value = []
    predict_result_dict = {}

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    train_loc = '../../production_data/tweeted_dataset_norm/train_data'
    test_loc = '../../production_data/tweeted_dataset_norm/test_data'

    train_matrix = []
    for subdir, _, files in os.walk(train_loc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                header = fin.readline()
                for line in fin:
                    row = np.zeros(1+2+category_cnt+lang_cnt+1)
                    _, publish, duration, definition, category, detect_lang, _, _, _, _, _, _, re30, _ = line.rstrip().split('\t', 13)
                    row[0] = np.log10(int(duration))
                    if definition == '0':
                        row[1] = 1
                    else:
                        row[2] = 1
                    row[3+category_dict[category]] = 1
                    row[3+category_cnt+lang_dict[detect_lang]] = 1
                    row[-1] = float(re30)
                    train_matrix.append(row)
    train_matrix = np.array(train_matrix)

    train_x = train_matrix[:, :-1]
    train_y = train_matrix[:, -1].reshape(-1, 1)

    test_matrix = []
    test_vids = []
    for subdir, _, files in os.walk(test_loc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                header = fin.readline()
                for line in fin:
                    row = np.zeros(1+2+category_cnt+lang_cnt)
                    vid, publish, duration, definition, category, detect_lang, _, _, _, _, _, _, re30, _ = line.rstrip().split('\t', 13)
                    row[0] = np.log10(int(duration))
                    if definition == '0':
                        row[1] = 1
                    else:
                        row[2] = 1
                    row[3+category_dict[category]] = 1
                    row[3+category_cnt+lang_dict[detect_lang]] = 1
                    test_matrix.append(row)
                    test_vids.append(vid)
                    true_value.append(float(re30))
    test_x = np.array(test_matrix)

    print('>>> Finish loading all data!')
    print('>>> Number of train observations: {0}'.format(train_x.shape[0]))
    print('>>> Number of train features: {0}'.format(train_x.shape[1]))
    print('>>> Number of test observations: {0}'.format(test_x.shape[0]))
    print('>>> Number of test features: {0}'.format(test_x.shape[1]))

    # ridge regression, log duration on relative engagement
    predict_value = ridge_regression(train_x, train_y, test_x)
    print('>>> Predict relative engagement on content with ridge regressor...')
    print('>>> MAE on test set: {0:.4f}'.format(mean_absolute_error(true_value, predict_value)))
    print('>>> R2 on test set: {0:.4f}'.format(r2_score(true_value, predict_value)))
    print('=' * 79)

    predict_result_dict = {vid: pred for vid, pred in zip(test_vids, predict_value)}

    # write to pickle file
    to_write = True
    if to_write:
        print('>>> Prepare to write to pickle file...')
        print('>>> Number of videos in final test result dict: {0}'.format(len(predict_result_dict)))
        write_dict_to_pickle(dict=predict_result_dict, path='./output/lr_content_predictor.p')
