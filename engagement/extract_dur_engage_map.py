#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os
import numpy as np
from collections import defaultdict
import cPickle as pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Extract global duration ~ watch percentage mapping from tweeted videos dataset

# 'id', 'publish', 'duration', 'definition', 'category', 'detect_lang', 'channel', 'topics', 'topics_num',
# 'view@30', 'watch@30', 'wp@30', 'view@120', 'watch@120', 'wp@120',
# 'days', 'daily_view', 'daily_watch'


def strify(iterable_struct):
    """
    Convert an iterable structure to comma separated string
    :param iterable_struct: an iterable structure
    :return: a string with comma separated
    """
    return ','.join(map(str, iterable_struct))


def get_duration_wp_from_file(filepath):
    with open(filepath, 'r') as fin:
        fin.readline()
        for line in fin:
            _, _, duration, dump = line.rstrip().split('\t', 3)
            _, _, _, _, _, _, _, _, wp30, _ = dump.split('\t', 9)
            duration = int(duration)
            wp30 = float(wp30)
            if wp30 > 1:
                wp30 = 1
            duration_wp_tuple.append((duration, wp30))
            duration_stats_dict[duration] += 1


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    categories = ['']
    bin_volume = 2000
    duration_wp_tuple = []
    duration_stats_dict = defaultdict(int)

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    input_loc = '../../data/production_data/tweeted_dataset'

    if os.path.isdir(input_loc):
        for subdir, _, files in os.walk(input_loc):
            for f in files:
                get_duration_wp_from_file(os.path.join(subdir, f))
                print('>>> Loading data: {0} done!'.format(f))
    else:
        get_duration_wp_from_file(input_loc)

    print('>>> Finish loading all data!')

    # sort by duration in ascent order
    sorted_duration_wp_tuple = sorted(duration_wp_tuple, key=lambda x: x[0])

    # get duration split point
    x_axis = []
    duration_list = sorted(duration_stats_dict.keys())
    freq_cnt = 0
    for dur in duration_list:
        freq_cnt += duration_stats_dict[dur]
        if freq_cnt > bin_volume:
            x_axis.append(dur)
            freq_cnt = 0
    if freq_cnt > 0:
        x_axis.append(duration_list[-1])

    # put videos in correct bins
    bin_matrix = []
    bin_list = []
    bin_idx = 0
    # put dur-wp tuple in the correct bin
    for item in sorted_duration_wp_tuple:
        if item[0] > x_axis[bin_idx]:
            bin_matrix.append(bin_list)
            bin_idx += 1
            bin_list = []
        bin_list.append(item[1])
    if len(bin_list) > 0:
        bin_matrix.append(bin_list)
    bin_matrix = [np.array(x) for x in bin_matrix]

    # sanity check
    to_check = True
    if to_check:
        print('videos in each bin')
        for i in xrange(len(x_axis)):
            print('duration split point: {0}; number of videos in bin: {1}'.format(x_axis[i], len(bin_matrix[i])))
        print('num of bins: {0}'.format(len(x_axis)))

    # store duration-engagement map as pickle file, each bin gets cut into 1000 percentiles
    to_pickle = True
    if to_pickle:
        dur_engage_map = {}
        dur_engage_map['duration'] = strify(x_axis)
        for idx, dur in enumerate(x_axis):
            dur_engage_bin = []
            for percentile in xrange(1, 1001):
                dur_engage_bin.append(np.percentile(bin_matrix[idx], percentile/10))
            dur_engage_map[idx] = strify(dur_engage_bin)
        pickle.dump(dur_engage_map, open('./data/tweeted_dur_engage_map.p', 'w'))

    # plot wp~dur distribution
    to_plot = True
    if to_plot:
        fig, ax1 = plt.subplots(1, 1)
        x_axis = np.log10(np.array(x_axis))
        ax1.plot(x_axis, [np.percentile(x, 1) for x in bin_matrix], 'g.-', label='9%', zorder=1)
        ax1.plot(x_axis, [np.percentile(x, 25) for x in bin_matrix], 'g--', label='25%', zorder=1)
        ax1.plot(x_axis, [np.mean(x) for x in bin_matrix], 'r-', label='Mean', zorder=1)
        ax1.plot(x_axis, [np.median(x) for x in bin_matrix], 'b--', label='Median', zorder=1)
        ax1.plot(x_axis, [np.percentile(x, 75) for x in bin_matrix], 'g--', label='75%', zorder=1)
        ax1.plot(x_axis, [np.percentile(x, 99) for x in bin_matrix], 'g.-', label='91%', zorder=1)

        def exponent(x, pos):
            'The two args are the value and tick position'
            return '%1.0f' % (10 ** x)

        x_formatter = FuncFormatter(exponent)
        ax1.xaxis.set_major_formatter(x_formatter)
        ax1.set_xlim([0, 5])
        ax1.set_ylim([0, 1])
        ax1.set_xlabel('Video duration (sec)', fontsize=20)
        ax1.set_ylabel('Watch percentage', fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=20)

        plt.tight_layout()
        plt.show()
