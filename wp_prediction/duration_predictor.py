#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Predict relative engagement from duration."""

from __future__ import print_function
import os
import cPickle as pickle
from sklearn.metrics import mean_absolute_error, r2_score


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    predict_result_dict = {}
    true_value = []
    predict_value = []

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    test_loc = '../../production_data/tweeted_dataset_norm/test_data'

    for subdir, _, files in os.walk(test_loc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                # read header
                fin.readline()
                for line in fin:
                    vid, dump = line.rstrip().split('\t', 1)
                    re30 = float(dump.split('\t')[11])
                    random_guess = 0.5
                    true_value.append(re30)
                    predict_value.append(random_guess)
                    predict_result_dict[vid] = random_guess

    print('>>> Predict relative engagement on duration...')
    print('>>> MAE on test set: {0:.4f}'.format(mean_absolute_error(true_value, predict_value)))
    print('>>> R2 on test set: {0:.4f}'.format(r2_score(true_value, predict_value)))
    print('=' * 79)

    # write to pickle file
    to_write = True
    if to_write:
        print('>>> Prepare to write to pickle file...')
        print('>>> Number of videos in final test result dict: {0}'.format(len(predict_result_dict)))
        output_path = './output/duration_predictor.p'
        folder_path = os.path.dirname(output_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        pickle.dump(predict_result_dict, open(output_path, 'wb'))
