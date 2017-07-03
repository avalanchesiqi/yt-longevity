#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import sys
import os
import json
import isodate
from datetime import datetime
from collections import defaultdict
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def read_as_int_array(content, truncated=None, delimiter=None):
    """
    Read input as an int array.
    :param content: string input
    :param truncated: head number of elements extracted
    :param delimiter: delimiter string
    :return: a numpy int array
    """
    if truncated is None:
        return np.array(map(int, content.split(delimiter)), dtype=np.uint32)
    else:
        return np.array(map(int, content.split(delimiter)[:truncated]), dtype=np.uint32)


def read_as_float_array(content, truncated=None, delimiter=None):
    """
    Read input as a float array.
    :param content: string input
    :param truncated: head number of elements extracted
    :param delimiter: delimiter string
    :return: a numpy float array
    """
    if truncated is None:
        return np.array(map(float, content.split(delimiter)), dtype=np.float64)
    else:
        return np.array(map(float, content.split(delimiter)[:truncated]), dtype=np.float64)


def get_percentile_boundaries(arr, bin_num):
    percentile_boundaries = [np.percentile(arr, 100/bin_num*bound) for bound in xrange(1, bin_num)]
    percentile_boundaries.append(np.inf)
    return np.array(percentile_boundaries)


def plot_five_number_summary(tuple_data, title_text=None):
    # get percentile boundaries
    percentile_boundaries = get_percentile_boundaries([x[0] for x in tuple_data], bin_num)

    # boxplot
    fig, ax1 = plt.subplots(1, 1)
    bin_matrix = [[] for _ in xrange(bin_num)]
    for x in tuple_data:
        bin_matrix[np.argmax(x[0] < percentile_boundaries)].append(x[1])
    print('>>> Plot figure: finish embedding into the matrix!')
    print([len(x) for x in bin_matrix])

    ax1.plot(np.arange(bin_num), [np.mean(bin_matrix[x]) for x in np.arange(bin_num)], 'ro-', label='Mean')
    ax1.plot(np.arange(bin_num), [np.median(bin_matrix[x]) for x in np.arange(bin_num)], 'bx--', label='Median')
    # ax1.boxplot(bin_matrix, showmeans=True, showfliers=False)

    # xticklabel
    xticklabels = []
    bin_width = 100/bin_num*5
    for i in xrange(bin_num):
        xticklabels.append('{0}%'.format(bin_width*i))
    ax1.set_xticklabels(xticklabels)
    for label in ax1.get_xaxis().get_ticklabels():
        label.set_visible(False)
    for label in ax1.get_xaxis().get_ticklabels()[::2]:
        label.set_visible(True)

    ax1.set_ylim(ymin=0)
    # ax1.set_ylim(ymax=1)
    ax1.set_xlabel('View Count (percentile)', fontsize=20)
    ax1.set_ylabel('Watch Time (minute)', fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    # fig.savefig(os.path.join(output_loc, 'week{0}.png'.format(week_idx + 1)))


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    bin_num = 40
    view_watch_tuple = []

    input_path = 'D:\GitWorks\data\production_data\\random_info'

    cnt = 0

    for subdir, _, files in os.walk(input_path):
        for f in files:
            with open(os.path.join(subdir, f)) as fin:
                fin.readline()
                for line in fin:
                    dump, views, watches = line.rstrip().rsplit(None, 2)
                    try:
                        views = read_as_int_array(views, delimiter=',')
                        watches = read_as_float_array(watches, delimiter=',')
                    except:
                        continue

                    view_num = np.sum(views)
                    if view_num == 0:
                        continue
                    avg_watch = np.sum(watches)/view_num
                    view_watch_tuple.append((view_num, avg_watch))
                    cnt += 1

                    # get total view number and avg watch time at 1st, 5th, 10th, 20th week
                    # for week in weeks:
                    #     view_num = np.sum(views[days < 7*week])
                    #     avg_watch = np.sum(watches[days < 7*week])/np.sum(views[days < 7*week])
            print('>>> Loading data: {0} done!'.format(f))
            # cnt += 1
            # if cnt > 0:
            #     break

    print('total video: {0}'.format(cnt))
    plot_five_number_summary(view_watch_tuple)
    plt.legend(numpoints=1, loc='lower right', fontsize=20)
    plt.show()
