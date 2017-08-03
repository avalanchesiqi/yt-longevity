#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os
import numpy as np
from collections import defaultdict
import cPickle as pickle
from scipy.stats import gaussian_kde
from matplotlib import gridspec
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
    bin_volume = 5000
    duration_wp_tuple = []
    duration_stats_dict = defaultdict(int)

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    input_loc = '../../data/tweeted_dataset_norm/test_data'

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
    to_pickle = False
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
        gs = gridspec.GridSpec(2, 2, width_ratios=[8, 1], height_ratios=[1, 8])
        fig = plt.figure(figsize=(9, 9))
        ax1 = plt.subplot(gs[1, 0])
        x_axis = np.log10(np.array(x_axis))

        plot_by_density = False
        if plot_by_density:
            density_kernel_y = []
            max_dense = 0
            for idx, single_bin in enumerate(bin_matrix[:-1]):
                density = gaussian_kde(single_bin)
                # generate a fake range of y values
                ys = np.arange(0, 1, .01)
                # fill y values using density class
                den = density(ys)
                if max(den) > max_dense:
                    max_dense = max(den)
            print('max dense', max_dense)

            for idx, single_bin in enumerate(bin_matrix[:-1]):
                density = gaussian_kde(single_bin)
                # generate a fake range of y values
                ys = np.arange(0, 1, .01)
                # fill y values using density class
                den = density(ys)
                for alpha_idx, alpha_value in enumerate(den[:-1]):
                    # print(idx, x_axis[idx], x_axis[idx+1], xs[alpha_idx], xs[alpha_idx+1])
                    ax1.fill_between([x_axis[idx], x_axis[idx+1]], ys[alpha_idx], ys[alpha_idx+1], alpha=den[alpha_idx]/max_dense, facecolor='k')

        for t in np.arange(5, 50, 5):
            ax1.fill_between(x_axis, [np.percentile(x, 50-t) for x in bin_matrix],
                             [np.percentile(x, 55-t) for x in bin_matrix], facecolor='b', alpha=(100-2*t)/100, lw=0)
            ax1.fill_between(x_axis, [np.percentile(x, 45+t) for x in bin_matrix],
                             [np.percentile(x, 50+t) for x in bin_matrix], facecolor='b', alpha=(100-2*t)/100, lw=0)

        def exponent(x, pos):
            'The two args are the value and tick position'
            return '%1.0f' % (10 ** x)

        x_formatter = FuncFormatter(exponent)
        ax1.xaxis.set_major_formatter(x_formatter)
        ax1.set_xlim([1, 5])
        ax1.set_ylim([0, 1])
        ax1.set_xlabel('Video duration (sec)', fontsize=20)
        ax1.set_ylabel('Watch percentage', fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        for label in ax1.get_xticklabels()[1::2]:
            label.set_visible(False)

        df_x = [np.log10(x[0]) for x in duration_wp_tuple]
        df_y = [x[1] for x in duration_wp_tuple]
        # KDE for top marginal
        kde_x = gaussian_kde(df_x)
        # KDE for right marginal
        kde_y = gaussian_kde(df_y)

        xmin, xmax = 1, 5
        ymin, ymax = 0, 1
        x = np.linspace(xmin, xmax, 100)
        y = np.linspace(ymin, ymax, 100)

        # Create Y-marginal (right)
        max_xlim = 1.2 * kde_y(y).max()
        axr = plt.subplot(gs[1, 1], xticks=[], yticks=[], frameon=False, xlim=(0, max_xlim), ylim=(ymin, ymax))
        axr.plot(kde_y(y), y, color='k')
        # axr.plot(kde_y[1](y), y, color='r')
        # axr.plot(kde_y[2](y), y, color='k')

        # Create X-marginal (top)
        max_ylim = 1.2 * kde_x(x).max()
        axt = plt.subplot(gs[0, 0], xticks=[], yticks=[], frameon=False, xlim=(xmin, xmax), ylim=(0, max_ylim))
        axt.plot(x, kde_x(x), color='k')
        # axt.plot(x, kde_x[1](x), color='r')
        # axt.plot(x, kde_x[2](x), color='k')

        # fig.tight_layout(pad=0)

        plt.show()
