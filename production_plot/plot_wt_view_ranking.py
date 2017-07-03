#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import sys
import os
import numpy as np
from scipy import stats
import operator
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


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    view_rank_dict = {}
    watch_rank_dict = {}

    input_path = 'D:\GitWorks\data\production_data\\random_info'

    cnt = 0

    for subdir, _, files in os.walk(input_path):
        for f in files:
            with open(os.path.join(subdir, f)) as fin:
                fin.readline()
                for line in fin:
                    dump, views, watches = line.rstrip().rsplit(None, 2)
                    id, _ = dump.split(None, 1)
                    try:
                        views = read_as_int_array(views, delimiter=',')
                        watches = read_as_float_array(watches, delimiter=',')
                    except:
                        continue

                    view_num = np.sum(views)
                    if view_num == 0:
                        continue

                    view_rank_dict[id] = view_num
                    watch_rank_dict[id] = np.sum(watches)

                    cnt += 1

                    # get total view number and avg watch time at 1st, 5th, 10th, 20th week
                    # for week in weeks:
                    #     view_num = np.sum(views[days < 7*week])
                    #     avg_watch = np.sum(watches[days < 7*week])/np.sum(views[days < 7*week])
            print('>>> Loading data: {0} done!'.format(f))
            # cnt += 1
            # if cnt > 0:
            #     break

    sorted_view_rank = sorted(view_rank_dict.items(), key=operator.itemgetter(1), reverse=True)[:10000]
    sorted_watch_rank = sorted(watch_rank_dict.items(), key=operator.itemgetter(1), reverse=True)[:10000]

    fig, ax1 = plt.subplots(1, 1)

    # iterate from 100 to 10000, with gap 100
    x_axis = []
    y_axis1 = []
    y_axis2 = []

    for i in xrange(100, 10001, 100):
        x_axis.append(i)

        view_set = set([item[0] for item in sorted_view_rank[:i]])
        watch_set = set([item[0] for item in sorted_watch_rank[:i]])

        num_common = len(list(view_set.intersection(watch_set))) / i
        y_axis1.append(num_common)

        unit_list = list(view_set.union(watch_set))
        view_rank = [view_rank_dict[x] for x in unit_list]
        watch_rank = [watch_rank_dict[x] for x in unit_list]

        y_axis2.append(stats.kendalltau(view_rank, watch_rank)[0])

    print('total video: {0}'.format(cnt))
    ax1.plot(x_axis, y_axis1, 'ro-', label='Common rate')
    ax1.plot(x_axis, y_axis2, 'bx--', label='Kendall tau')
    ax1.set_xlabel('Top n videos', fontsize=20)
    ax1.set_ylabel('Measurement', fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=20)

    plt.legend(numpoints=1, loc='upper right', fontsize=20)
    plt.tight_layout()
    plt.show()
