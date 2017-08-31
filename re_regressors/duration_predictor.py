#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Predict relative engagement from duration."""

from __future__ import print_function
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import time
from sklearn.metrics import mean_absolute_error, r2_score

from utils.helper import write_dict_to_pickle


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    start_time = time.time()

    test_vids = []
    true_re = []
    guess_re = []

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    test_loc = '../../production_data/tweeted_dataset_norm/test_data'

    for subdir, _, files in os.walk(test_loc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                # read header
                fin.readline()
                for line in fin:
                    vid, dump = line.rstrip().split('\t', 1)
                    test_vids.append(vid)
                    re30 = float(dump.split('\t')[11])
                    true_re.append(re30)
                    random_guess = 0.5
                    guess_re.append(random_guess)

    print('>>> Predict relative engagement on duration...')
    print('>>> MAE on test set: {0:.4f}'.format(mean_absolute_error(true_re, guess_re)))
    print('>>> R2 on test set: {0:.4f}'.format(r2_score(true_re, guess_re)))
    print('=' * 79)

    # get running time
    print('\n>>> Total running time: {0:.4f}'.format(time.time() - start_time))

    # write to pickle file
    to_write = True
    true_result_dict = {vid: true for vid, true in zip(test_vids, true_re)}
    predict_result_dict = {vid: pred for vid, pred in zip(test_vids, guess_re)}
    if to_write:
        print('>>> Prepare to write to pickle file...')
        print('>>> Number of videos in final test result dict: {0}'.format(len(test_vids)))
        write_dict_to_pickle(dict=true_result_dict, path='./output/true_predictor.p')
        write_dict_to_pickle(dict=predict_result_dict, path='./output/duration_predictor.p')
