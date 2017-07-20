#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# Extract global watch perc ~ duration bivariate from training part (13M) of a large collection of YouTube videos, 19M dataset


def strify(iterable_struct):
    """
    Convert an iterable structure to comma separated string
    :param iterable_struct: an iterable structure
    :return: a string with comma separated
    """
    return ','.join(map(str, iterable_struct))


def get_duration_wp_from_file(filepath):
    with open(filepath, 'r') as fin:
        for line in fin:
            vid, duration, definition, category_id, channel_id, topics, total_view, true_wp = line.rstrip().split('\t')
            duration = int(duration)
            true_wp = float(true_wp)
            duration_wp_tuple.append((duration, true_wp))
            duration_stats_dict[duration] += 1


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    bin_volume = 3
    duration_wp_tuple = []
    duration_stats_dict = defaultdict(int)

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    input_path = '../../data/production_data/wp_prediction/train_data/trailer.txt'

    if os.path.isdir(input_path):
        for subdir, _, files in os.walk(input_path):
            for f in files:
                get_duration_wp_from_file(os.path.join(subdir, f))
                print('>>> Loading data: {0} done!'.format(f))
    else:
        get_duration_wp_from_file(input_path)

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
    to_check = False
    if to_check:
        print('videos in each bin')
        for i in xrange(len(x_axis)):
            print('duration split point: {0}; number of videos in bin: {1}'.format(x_axis[i], len(bin_matrix[i])))

    # write to disk file
    to_write = True
    if to_write:
        upper_axis = [np.percentile(x, 99) for x in bin_matrix]
        mean_axis = [np.mean(x) for x in bin_matrix]
        median_axis = [np.median(x) for x in bin_matrix]
        lower_axis = [np.percentile(x, 1) for x in bin_matrix]

        print(len(mean_axis))

        with open('global_params/global_parameters_trailer.txt', 'w') as fout:
            fout.write('{0}\n'.format(strify(x_axis)))
            fout.write('{0}\n'.format(strify(mean_axis)))
            fout.write('{0}\n'.format(strify(median_axis)))
            fout.write('{0}\n'.format(strify(upper_axis)))
            fout.write('{0}\n'.format(strify(lower_axis)))

    # check distribution in each bin
    to_plot1 = False
    if to_plot1:
        fig = plt.figure()
        shown_bin = [40, 70, 100, 130, 160, 100, 240, 280, 320]
        for i in xrange(9):
            ax = fig.add_subplot(331+i)
            ax.hist(bin_matrix[shown_bin[i]], bins=50)

    # plot wp~dur distribution
    to_plot2 = True
    if to_plot2:
        fig, ax1 = plt.subplots(1, 1)
        ax1.plot(x_axis, [np.percentile(x, 1) for x in bin_matrix], 'g.-', label='1%', zorder=1)
        ax1.plot(x_axis, [np.percentile(x, 25) for x in bin_matrix], 'g--', label='25%', zorder=1)
        ax1.plot(x_axis, [np.mean(x) for x in bin_matrix], 'r-', label='Mean', zorder=1)
        ax1.plot(x_axis, [np.median(x) for x in bin_matrix], 'b--', label='Median', zorder=1)
        ax1.plot(x_axis, [np.percentile(x, 75) for x in bin_matrix], 'g--', label='75%', zorder=1)
        ax1.plot(x_axis, [np.percentile(x, 99) for x in bin_matrix], 'g.-', label='99%', zorder=1)

        ax1.set_xlim(xmin=x_axis[0])
        ax1.set_ylim([0, 1])
        ax1.set_xlabel('Video duration (sec)', fontsize=20)
        ax1.set_ylabel('Watch percentage', fontsize=20)
        ax1.set_xscale('log')
        ax1.tick_params(axis='both', which='major', labelsize=20)

    # plt.legend(numpoints=1, loc='upper right', fontsize=20)
    plt.tight_layout()
    plt.show()
