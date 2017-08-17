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


def remove_bad_bins(x_axis, bin_matrix):
    x_axis = x_axis[:len(bin_matrix)]
    bad_bin_idx = []
    for idx, bin in enumerate(bin_matrix):
        if len(bin) < 30:
            bad_bin_idx.append(idx)
    for idx in bad_bin_idx[::-1]:
        x_axis.pop(idx)
        bin_matrix.pop(idx)
    return x_axis, bin_matrix


def plot_contour(x_axis_value, color='r', fsize=14, title=False):
    target_bin = bin_matrix[np.sum(x_axis < x_axis_value)]
    ax1.plot([x_axis_value, x_axis_value], [np.percentile(target_bin, 0.5), np.percentile(target_bin, 99.5)], c=color, zorder=20)
    for t in xrange(10, 95, 10):
        ax1.plot([x_axis_value - 0.04, x_axis_value + 0.04],
                 [np.percentile(target_bin, t), np.percentile(target_bin, t)], c=color, zorder=20)
        if t%20 == 0:
            ax1.text(x_axis_value + 0.1, np.percentile(target_bin, t), '{0}%'.format(int(t)), fontsize=fsize, verticalalignment='center', zorder=30)
    for t in [0.5, 99.5]:
        ax1.plot([x_axis_value - 0.04, x_axis_value + 0.04],
                 [np.percentile(target_bin, t), np.percentile(target_bin, t)], c=color, zorder=20)
    if title:
        ax1.text(x_axis_value, np.percentile(target_bin, 99.5)+0.02, r'$\bar \eta_{30}$', color='k', fontsize=28, horizontalalignment='center', zorder=30)


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    categories = ['']
    bin_number = 500
    duration_wp_tuple = []
    duration_stats_dict = defaultdict(int)

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    input_loc = '../../data/production_data/tweeted_dataset_norm'

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
    duration_list = sorted(duration_stats_dict.keys())
    x_axis = list(np.linspace(1, 5, bin_number))

    # put videos in correct bins
    bin_matrix = []
    bin_list = []
    bin_idx = 0
    # put dur-wp tuple in the correct bin
    for item in sorted_duration_wp_tuple:
        if np.log10(item[0]) > x_axis[bin_idx]:
            bin_matrix.append(bin_list)
            bin_idx += 1
            bin_list = []
        bin_list.append(item[1])
    if len(bin_list) > 0:
        bin_matrix.append(bin_list)
    bin_matrix = [np.array(x) for x in bin_matrix]

    x_axis, bin_matrix = remove_bad_bins(x_axis, bin_matrix)

    # sanity check
    to_check = True
    if to_check:
        print('videos in each bin')
        for i in xrange(len(bin_matrix)):
            print('duration split point: {0}; number of videos in bin: {1}'.format(x_axis[i], len(bin_matrix[i])))
        print('num of bins: {0}'.format(len(bin_matrix)))

    # plot wp~dur distribution
    to_plot = True
    if to_plot:
        gs = gridspec.GridSpec(2, 2, width_ratios=[8, 1], height_ratios=[1, 8])
        # gs.update(wspace=0.025, hspace=0.025)
        fig = plt.figure(figsize=(9, 9))
        ax1 = plt.subplot(gs[1, 0])
        # x_axis = np.log10(np.array(x_axis))

        for t in np.arange(5, 50, 5):
            ax1.fill_between(x_axis, [np.percentile(x, 50-t) for x in bin_matrix],
                             [np.percentile(x, 55-t) for x in bin_matrix], facecolor='#6495ed', alpha=(100-2*t)/100, lw=0)
            ax1.fill_between(x_axis, [np.percentile(x, 45+t) for x in bin_matrix],
                             [np.percentile(x, 50+t) for x in bin_matrix], facecolor='#6495ed', alpha=(100-2*t)/100, lw=0)

        for t in [10, 30, 70, 90]:
            ax1.plot(x_axis, [np.percentile(x, t) for x in bin_matrix], color='#6495ed', alpha=0.8, zorder=15)
        ax1.plot(x_axis, [np.percentile(x, 50) for x in bin_matrix], color='#6495ed', alpha=1, zorder=15)

        def exponent(x, pos):
            'The two args are the value and tick position'
            return '%1.0f' % (10 ** x)

        x_formatter = FuncFormatter(exponent)
        ax1.xaxis.set_major_formatter(x_formatter)
        ax1.set_xlim([1, 5])
        ax1.set_ylim([0, 1])
        ax1.set_xlabel('video duration (sec) '+r'$D$', fontsize=24)
        ax1.set_ylabel('watch percentage '+r'$\bar \mu_{30}$', fontsize=24)
        ax1.tick_params(axis='both', which='major', labelsize=24)
        for label in ax1.get_xticklabels()[1::2]:
            label.set_visible(False)

        # ax1.legend([plt.Rectangle((0, 0), 1, 1, fc='#6495ed')], ['Tweeted Videos'], loc='upper right', fontsize=20)

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
        axr.plot(kde_y(y), y, color='#6495ed')

        # Create X-marginal (top)
        max_ylim = 1.2 * kde_x(x).max()
        axt = plt.subplot(gs[0, 0], xticks=[], yticks=[], frameon=False, xlim=(xmin, xmax), ylim=(0, max_ylim))
        axt.plot(x, kde_x(x), color='#6495ed')

        plot_examples = True
        if plot_examples:
            # d_8ao3o5ohU, Black Belt Kid Vs. White Belt Adults, 6309812
            quality_short = (287, 0.7022605, '$\mathregular{V_{1}}$: d_8ao3o5ohU')
            # akuyBBIbOso, Learn Colors with Squishy Mesh Balls for Toddlers Kids and Children - Surprise Eggs for Babies, 6449735
            junk_short = (306, 0.2066883, '$\mathregular{V_{2}}$: akuyBBIbOso')
            # WH7llf2vaKQ, Joe Rogan Experience - Fight Companion - August 6, 2016, 490585
            quality_long = (13779, 0.1900219, '$\mathregular{V_{3}}$: WH7llf2vaKQ')

            points = [quality_short, junk_short, quality_long]
            for point in points:
                ax1.scatter(np.log10(point[0]), point[1], c='#ff4500', s=30, lw=1)
                ax1.text(np.log10(point[0]), point[1]+0.02, point[2],
                         horizontalalignment='center', size=20, color='k', zorder=25)

            plot_contour((np.log10(287) + np.log10(306)) / 2, color='k', fsize=18, title=True)
            plot_contour(np.log10(13779), color='k', fsize=14)

        fig.tight_layout(pad=0.2, h_pad=0.02, w_pad=0.001)
        plt.show()
