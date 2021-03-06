#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Predict watch percentage from channel specific predictor, with ridge regression.

Time: ~20M
"""

from __future__ import division, print_function
import os, sys, time, datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
from collections import defaultdict

from utils.helper import write_dict_to_pickle
from utils.ridge_regressor import RidgeRegressor


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    start_time = time.time()
    predict_result_dict = {}
    k = 5

    category_dict = {'1': 0, '2': 1, '10': 2, '15': 3, '17': 4, '19': 5, '20': 6, '22': 7, '23': 8, '24': 9,
                     '25': 10, '26': 11, '27': 12, '28': 13, '29': 14, '30': 15, '43': 16, '44': 17}
    category_cnt = len(category_dict)

    lang_dict = {'af': 0, 'ar': 1, 'bg': 2, 'bn': 3, 'ca': 4, 'cs': 5, 'cy': 6, 'da': 7, 'de': 8, 'el': 9, 'en': 10,
                 'es': 11, 'et': 12, 'fa': 13, 'fi': 14, 'fr': 15, 'gu': 16, 'he': 17, 'hi': 18, 'hr': 19, 'hu': 20,
                 'id': 21, 'it': 22, 'ja': 23, 'kn': 24, 'ko': 25, 'lt': 26, 'lv': 27, 'mk': 28, 'ml': 29, 'mr': 30,
                 'ne': 31, 'nl': 32, 'no': 33, 'pa': 34, 'pl': 35, 'pt': 36, 'ro': 37, 'ru': 38, 'sk': 39, 'sl': 40,
                 'so': 41, 'sq': 42, 'sv': 43, 'sw': 44, 'ta': 45, 'te': 46, 'th': 47, 'tl': 48, 'tr': 49, 'uk': 50,
                 'ur': 51, 'vi': 52, 'zh-cn': 53, 'zh-tw': 54, 'NA': 55}
    lang_cnt = len(lang_dict)

    train_channel_cnt_map = defaultdict(int)
    with open('./data/train_channel_watch_percentage.txt', 'r') as fin:
        for line in fin:
            channel, wp30 = line.rstrip().split('\t')
            train_channel_cnt_map[channel] += 1

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    data_loc = '../../production_data/new_tweeted_channel_dataset'
    train_loc = os.path.join(data_loc, 'train_data')
    test_loc = os.path.join(data_loc, 'test_data')

    # == == == == == == == == Part 3: Start training == == == == == == == == #
    for subdir, _, files in os.walk(test_loc):
        for f in files:
            # if we have observed this channel before, minimal observations: k
            if f in train_channel_cnt_map and train_channel_cnt_map[f] >= k:
                sub_f = f[:4]
                train_data_path = os.path.join(train_loc, sub_f, f)
                train_lines = open(train_data_path, 'r').readlines()

                # get topic encoding
                topic_dict = {'NA': 0}
                topic_cnt = 1
                for topics in [x.split('\t')[7] for x in train_lines]:
                    if (not topics == '') and (not topics == 'NA'):
                        for topic in topics.split(','):
                            if topic not in topic_dict:
                                topic_dict[topic] = topic_cnt
                                topic_cnt += 1

                # get channel history
                train_matrix = []
                with open(train_data_path, 'r') as fin:
                    for line in fin:
                        row = np.zeros(1+2+category_cnt+lang_cnt+topic_cnt+1)
                        vid, publish, duration, definition, category, detect_lang, channel, topics, _, _, wp30, _, _ = line.rstrip().split('\t', 12)
                        row[0] = np.log10(int(duration))
                        if definition == '0':
                            row[1] = 1
                        else:
                            row[2] = 1
                        row[3+category_dict[category]] = 1
                        row[3+category_cnt+lang_dict[detect_lang]] = 1
                        if topics == '' or topics == 'NA':
                            row[3 + category_cnt + lang_cnt + topic_dict['NA']] = 1
                        else:
                            topics = topics.split(',')
                            for topic in topics:
                                row[3+category_cnt+lang_cnt+topic_dict[topic]] = 1
                        row[-1] = float(wp30)
                        train_matrix.append(row)
                train_matrix = np.array(train_matrix)

                test_matrix = []
                test_vids = []
                with open(os.path.join(subdir, f), 'r') as fin:
                    for line in fin:
                        row = np.zeros(1+2+category_cnt+lang_cnt+topic_cnt+1)
                        vid, publish, duration, definition, category, detect_lang, channel, topics, _, _, wp30, _, _ = line.rstrip().split('\t', 12)
                        row[0] = np.log10(int(duration))
                        if definition == '0':
                            row[1] = 1
                        else:
                            row[2] = 1
                        if category in category_dict:
                            row[3+category_dict[category]] = 1
                        if detect_lang in lang_dict:
                            row[3+category_cnt + lang_dict[detect_lang]] = 1
                        if topics == '' or topics == 'NA':
                            row[3 + category_cnt + lang_cnt + topic_dict['NA']] = 1
                        else:
                            topics = topics.split(',')
                            for topic in topics:
                                if topic in topic_dict:
                                    row[3 + category_cnt + lang_cnt + topic_dict[topic]] = 1
                                else:
                                    row[3 + category_cnt + lang_cnt + topic_dict['NA']] = 1
                        row[-1] = float(wp30)
                        test_matrix.append(row)
                        test_vids.append(vid)
                test_matrix = np.array(test_matrix)

                # predict test data from customized ridge regressor
                test_yhat = RidgeRegressor(train_matrix, test_matrix, verbose=False).predict()

                predict_result_dict.update({vid: pred for vid, pred in zip(test_vids, test_yhat)})

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])

    # write to pickle file
    to_write = True
    if to_write:
        print('>>> Prepare to write to pickle file...')
        print('>>> Number of videos in final test result dict: {0}'.format(len(predict_result_dict)))
        write_dict_to_pickle(dict=predict_result_dict, path='./output/csp_predictor_{0}.p'.format(k))
