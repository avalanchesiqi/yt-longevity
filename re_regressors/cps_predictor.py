#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Predict relative engagement from channel past success, with ridge regression."""

from __future__ import division, print_function
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from collections import defaultdict
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
    true_value = []
    predict_value = []
    predict_result_dict = {}

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    channel_re_dict = defaultdict(list)
    with open('../engagement/data/train_channel_es.txt', 'r') as fin:
        for line in fin:
            channel, re30 = line.rstrip().split('\t')
            channel_re_dict[channel].append(float(re30))

    train_loc = '../../production_data/tweeted_dataset_norm/train_data'
    test_loc = '../../production_data/tweeted_dataset_norm/test_data'

    train_matrix = []
    for subdir, _, files in os.walk(train_loc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                header = fin.readline()
                for line in fin:
                    row = np.zeros(9)
                    _, _, duration, _, _, _, channel, _, _, _, _, _, re30, _ = line.rstrip().split('\t', 13)
                    row[0] = np.log10(int(duration))
                    if channel in channel_re_dict:
                        row[1] = np.mean(channel_re_dict[channel])
                        row[2] = np.std(channel_re_dict[channel])
                        row[3] = np.min(channel_re_dict[channel])
                        row[4] = np.percentile(channel_re_dict[channel], 25)
                        row[5] = np.median(channel_re_dict[channel])
                        row[6] = np.percentile(channel_re_dict[channel], 75)
                        row[7] = np.max(channel_re_dict[channel])
                    row[-1] = float(re30)
                    train_matrix.append(row)
            print('>>> Finish loading file {0}!'.format(f))
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
                    row = np.zeros(8)
                    vid, _, duration, _, _, _, channel, _, _, _, _, _, re30, _ = line.rstrip().split('\t', 13)
                    row[0] = np.log10(int(duration))
                    if channel in channel_re_dict:
                        row[1] = np.mean(channel_re_dict[channel])
                        row[2] = np.std(channel_re_dict[channel])
                        row[3] = np.min(channel_re_dict[channel])
                        row[4] = np.percentile(channel_re_dict[channel], 25)
                        row[5] = np.median(channel_re_dict[channel])
                        row[6] = np.percentile(channel_re_dict[channel], 75)
                        row[7] = np.max(channel_re_dict[channel])
                    test_matrix.append(row)
                    test_vids.append(vid)
                    true_value.append(float(re30))
            print('>>> Finish loading file {0}!'.format(f))
    test_x = np.array(test_matrix)

    print('>>> Finish loading all data!')
    print('>>> Number of train observations: {0}'.format(train_x.shape[0]))
    print('>>> Number of train features: {0}'.format(train_x.shape[1]))
    print('>>> Number of test observations: {0}'.format(test_x.shape[0]))
    print('>>> Number of test features: {0}'.format(test_x.shape[1]))

    print(test_matrix[:5, :])

    # ridge regression, log duration on relative engagement
    predict_value = ridge_regression(train_x, train_y, test_x)
    print('>>> Predict relative engagement on channel past success with ridge regressor...')
    print('>>> MAE on test set: {0:.4f}'.format(mean_absolute_error(true_value, predict_value)))
    print('>>> R2 on test set: {0:.4f}'.format(r2_score(true_value, predict_value)))
    print('=' * 79)

    predict_result_dict = {vid: pred for vid, pred in zip(test_vids, predict_value)}

    # write to pickle file
    to_write = True
    if to_write:
        print('>>> Prepare to write to pickle file...')
        print('>>> Number of videos in final test result dict: {0}'.format(len(predict_result_dict)))
        write_dict_to_pickle(dict=predict_result_dict, path='./output/cps_predictor.p')
