#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import cPickle as pickle

# Predict watch percentage from duration only


def get_wp(duration, percentile):
    bin_idx = np.sum(duration_split_points < duration)
    duration_bin = dur_engage_map[bin_idx]
    percentile = int(round(percentile*1000))
    wp_percentile = duration_bin[percentile]
    return wp_percentile


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    dur_engage_str_map = pickle.load(open('dur_engage_map.p', 'rb'))
    dur_engage_map = {key: list(map(float, value.split(','))) for key, value in dur_engage_str_map.items()}

    duration_split_points = np.array(dur_engage_map['duration'])

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    input_loc = '../../data/production_data/random_norm/test_data'
    predict_result_dict = {}

    for subdir, _, files in os.walk(input_loc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                # read header
                fin.readline()
                for line in fin:
                    vid, duration, _ = line.rstrip().split('\t', 2)
                    duration = int(duration)
                    median_percentile = 0.5
                    dur_wp = get_wp(duration, median_percentile)
                    predict_result_dict[vid] = dur_wp

    # write to pickle file
    to_write = True
    if to_write:
        output_path = 'norm_predict_results/predict_d.p'
        print('>>> Number of videos in final test result dict: {0}'.format(len(predict_result_dict)))
        pickle.dump(predict_result_dict, open(output_path, 'wb'))
