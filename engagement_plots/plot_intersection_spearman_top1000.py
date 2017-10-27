#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Scripts to plot Figure 2, co-occurrence and spearman rho for top 1000 videos in terms of views and watch times.

Usage: python plot_intersection_spearman_top1000.py
Time: ~1M
"""

from __future__ import division, print_function
import os
import numpy as np
from scipy import stats
import operator
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    view_rank_dict = {}
    watch_rank_dict = {}

    input_doc = '../../production_data/new_tweeted_dataset_norm/'

    for subdir, _, files in os.walk(input_doc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                fin.readline()
                for line in fin:
                    vid, dump = line.rstrip().split('\t', 1)[0]
                    view30 = float(dump.split('\t')[7])
                    watch30 = float(dump.split('\t')[8])

                    view_rank_dict[vid] = view30
                    watch_rank_dict[vid] = watch30
            print('>>> Loading data: {0} done!'.format(f))

    sorted_view_rank = sorted(view_rank_dict.items(), key=operator.itemgetter(1), reverse=True)[:1000]
    sorted_watch_rank = sorted(watch_rank_dict.items(), key=operator.itemgetter(1), reverse=True)[:1000]

    # iterate from 10 to 1000, with gap 10
    x_axis = []
    y_axis1 = []
    y_axis2 = []

    for i in range(10, 1001, 10):
        view_set = set([item[0] for item in sorted_view_rank[:i]])
        watch_set = set([item[0] for item in sorted_watch_rank[:i]])

        unit_list = list(view_set.union(watch_set))
        view_rank = [view_rank_dict[x] for x in unit_list]
        watch_rank = [watch_rank_dict[x] for x in unit_list]

        x_axis.append(i)
        y_axis1.append(len(view_set.intersection(watch_set))/i)
        y_axis2.append(stats.spearmanr(view_rank, watch_rank)[0])

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(x_axis, y_axis1)
    ax1.set_xlabel('Top n videos', fontsize=16)
    ax1.set_ylabel('Intersection', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.plot(x_axis, y_axis2)
    ax2.set_xlabel('Top n videos', fontsize=16)
    ax2.set_ylabel("Spearman's $\\rho$", fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.show()
