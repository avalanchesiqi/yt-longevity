#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Predict watch percentage from duration."""

from __future__ import print_function
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import time, datetime
import cPickle as pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from utils.helper import write_dict_to_pickle
from utils.converter import to_watch_percentage


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    start_time = time.time()

    dur_engage_str_map = pickle.load(open('./data/tweeted_dur_engage_map.p', 'rb'))
    dur_engage_map = {key: list(map(float, value.split(','))) for key, value in dur_engage_str_map.items()}
    lookup_durations = np.array(dur_engage_map['duration'])

    test_vids = []
    true_wp = []
    guess_wp = []

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    test_loc = '../../production_data/tweeted_dataset_norm/test_data'

    for subdir, _, files in os.walk(test_loc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                # read header
                fin.readline()
                for line in fin:
                    vid, _, duration, dump = line.rstrip().split('\t', 3)
                    test_vids.append(vid)
                    duration = int(duration)
                    wp30 = float(dump.split('\t')[8])
                    true_wp.append(wp30)
                    random_guess = 0.5
                    guess_wp.append(to_watch_percentage(dur_engage_map, duration, random_guess, lookup_keys=lookup_durations))

    print('>>> Predict watch percentage on duration...')
    print('>>> MAE on test set: {0:.4f}'.format(mean_absolute_error(true_wp, guess_wp)))
    print('>>> R2 on test set: {0:.4f}'.format(r2_score(true_wp, guess_wp)))
    print('=' * 79)

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])

    # write to pickle file
    to_write = True
    true_result_dict = {vid: true for vid, true in zip(test_vids, true_wp)}
    predict_result_dict = {vid: pred for vid, pred in zip(test_vids, guess_wp)}
    if to_write:
        print('>>> Prepare to write to pickle file...')
        print('>>> Number of videos in final test result dict: {0}'.format(len(test_vids)))
        write_dict_to_pickle(dict=true_result_dict, path='./output/true_predictor.p')
        write_dict_to_pickle(dict=predict_result_dict, path='./output/duration_predictor.p')
