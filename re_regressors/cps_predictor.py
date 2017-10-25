#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Predict relative engagement from channel past success, with ridge regression."""

from __future__ import division, print_function
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import time, datetime
from collections import defaultdict
import numpy as np

from utils.helper import write_dict_to_pickle
from utils.ridge_regressor import RidgeRegressor


def _load_data(filepath):
    """Load features space for channel past success predictor."""
    matrix = []
    vids = []
    with open(filepath, 'r') as fin:
        fin.readline()
        for line in fin:
            row = np.zeros(10)
            vid, _, duration, _, _, _, channel, _, _, _, _, re30, _ = line.rstrip().split('\t', 12)
            if channel in channel_re_dict:
                row[0] = np.log10(int(duration))
                row[1] = len(channel_re_dict[channel]) / 52
                row[2] = np.mean(channel_re_dict[channel])
                row[3] = np.std(channel_re_dict[channel])
                row[4] = np.min(channel_re_dict[channel])
                row[5] = np.percentile(channel_re_dict[channel], 25)
                row[6] = np.median(channel_re_dict[channel])
                row[7] = np.percentile(channel_re_dict[channel], 75)
                row[8] = np.max(channel_re_dict[channel])
                row[9] = float(re30)
                matrix.append(row)
                vids.append(vid)
    print('>>> Finish loading file {0}!'.format(filepath))
    return matrix, vids


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    start_time = time.time()

    channel_re_dict = defaultdict(list)
    with open('./data/train_channel_relative_engagement.txt', 'r') as fin:
        for line in fin:
            channel, re30 = line.rstrip().split('\t')
            channel_re_dict[channel].append(float(re30))

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    data_loc = '../../production_data/tweeted_dataset_norm'
    train_loc = os.path.join(data_loc, 'train_data')
    test_loc = os.path.join(data_loc, 'test_data')

    print('>>> Start to load training dataset...')
    train_matrix = []
    for subdir, _, files in os.walk(train_loc):
        for f in files:
            train_matrix.extend(_load_data(os.path.join(subdir, f))[0])
    train_matrix = np.array(train_matrix)

    print('>>> Start to load test dataset...')
    test_matrix = []
    test_vids = []
    for subdir, _, files in os.walk(test_loc):
        for f in files:
            matrix, vids = _load_data(os.path.join(subdir, f))
            test_matrix.extend(matrix)
            test_vids.extend(vids)
    test_matrix = np.array(test_matrix)

    print('>>> Finish loading all data!\n')

    # predict test data from customized ridge regressor
    test_yhat = RidgeRegressor(train_matrix, test_matrix).predict()

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])

    # write to pickle file
    to_write = True
    predict_result_dict = {vid: pred for vid, pred in zip(test_vids, test_yhat)}
    if to_write:
        print('>>> Prepare to write to pickle file...')
        print('>>> Number of videos in final test result dict: {0}'.format(len(predict_result_dict)))
        write_dict_to_pickle(dict=predict_result_dict, path='./output/cps_predictor.p')
