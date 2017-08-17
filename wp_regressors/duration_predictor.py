#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Predict watch percentage from duration."""

from __future__ import print_function
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
import cPickle as pickle
from sklearn.metrics import mean_absolute_error, r2_score

from utils.helper import write_dict_to_pickle


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    dur_engage_str_map = pickle.load(open('../engagement/data/tweeted_dur_engage_map.p', 'rb'))
    dur_engage_map = {key: list(map(float, value.split(','))) for key, value in dur_engage_str_map.items()}
    lookup_durations = np.array(dur_engage_map['duration'])
    true_value = []
    predict_value = []
    predict_result_dict = {}

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    test_loc = '../../production_data/tweeted_dataset_norm/test_data'

    for subdir, _, files in os.walk(test_loc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                # read header
                fin.readline()
                for line in fin:
                    vid, _, duration, dump = line.rstrip().split('\t', 3)
                    wp30 = float(dump.split('\t')[8])
                    duration = int(duration)

                    true_value.append(wp30)
                    random_guess = np.mean(dur_engage_map[np.sum(lookup_durations < duration)])
                    predict_value.append(random_guess)
                    predict_result_dict[vid] = random_guess

    print('>>> Predict watch percentage on duration...')
    print('>>> MAE on test set: {0:.4f}'.format(mean_absolute_error(true_value, predict_value)))
    print('>>> R2 on test set: {0:.4f}'.format(r2_score(true_value, predict_value)))
    print('=' * 79)

    # write to pickle file
    to_write = True
    if to_write:
        print('>>> Prepare to write to pickle file...')
        print('>>> Number of videos in final test result dict: {0}'.format(len(predict_result_dict)))
        write_dict_to_pickle(dict=predict_result_dict, path='./output/duration_predictor.p')
