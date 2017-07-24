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

    output_file = open('norm_predict_results/predict_d.txt', 'w')

    to_write_header = True
    for subdir, _, files in os.walk(input_loc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                header = fin.readline().rstrip()
                if to_write_header:
                    output_file.write('{0}\t{1}\n'.format(header, 'd_wp'))
                    to_write_header = False
                for line in fin:
                    _, duration, _ = line.rstrip().split('\t', 2)
                    duration = int(duration)
                    median_percentile = 0.5
                    dur_wp = get_wp(duration, median_percentile)
                    output_file.write('{0}\t{1}\n'.format(line.rstrip(), dur_wp))
    output_file.close()
