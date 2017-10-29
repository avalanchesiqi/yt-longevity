#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Scripts to plot Figure 2, co-occurrence and spearman rho for top 1000 videos in terms of views and watch times.

Usage: python plot_intersection_spearman_top1000.py
Time: ~2M
"""

from __future__ import division, print_function
import os
from scipy import stats
import operator
import matplotlib.pyplot as plt


def plot_intersection(ax, view_rank_dict, watch_rank_dict, color, linestyle, label):
    sorted_view_rank = sorted(view_rank_dict.items(), key=operator.itemgetter(1), reverse=True)[:1000]
    sorted_watch_rank = sorted(watch_rank_dict.items(), key=operator.itemgetter(1), reverse=True)[:1000]

    x_axis = []
    y_axis = []
    # iterate from 50 to 1000, with gap 10
    for i in range(50, 1001, 10):
        view_set = set([item[0] for item in sorted_view_rank[:i]])
        watch_set = set([item[0] for item in sorted_watch_rank[:i]])
        x_axis.append(i)
        y_axis.append(len(view_set.intersection(watch_set))/i)

    ax.plot(x_axis, y_axis, color=color, linestyle=linestyle, label=label)


def plot_spearman(ax, view_rank_dict, watch_rank_dict, color, linestyle, label):
    sorted_view_rank = sorted(view_rank_dict.items(), key=operator.itemgetter(1), reverse=True)[:1000]
    sorted_watch_rank = sorted(watch_rank_dict.items(), key=operator.itemgetter(1), reverse=True)[:1000]

    x_axis = []
    y_axis = []
    # iterate from 50 to 1000, with gap 10
    for i in range(50, 1001, 10):
        view_set = set([item[0] for item in sorted_view_rank[:i]])
        watch_set = set([item[0] for item in sorted_watch_rank[:i]])
        unit_list = list(view_set.union(watch_set))
        view_rank = [view_rank_dict[x] for x in unit_list]
        watch_rank = [watch_rank_dict[x] for x in unit_list]

        x_axis.append(i)
        y_axis.append(stats.spearmanr(view_rank, watch_rank)[0])

    ax.plot(x_axis, y_axis, color=color, linestyle=linestyle, label=label)


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    view_rank_dict = {}
    watch_rank_dict = {}
    music_view_rank_dict = {}
    music_watch_rank_dict = {}
    news_view_rank_dict = {}
    news_watch_rank_dict = {}

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    input_doc = '../../production_data/new_tweeted_dataset_norm/'
    for subdir, _, files in os.walk(input_doc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                fin.readline()
                for line in fin:
                    vid, dump = line.rstrip().split('\t', 1)
                    view30 = float(dump.split('\t')[7])
                    watch30 = float(dump.split('\t')[8])

                    view_rank_dict[vid] = view30
                    watch_rank_dict[vid] = watch30

                    if f.startswith('10'):
                        music_view_rank_dict[vid] = view30
                        music_watch_rank_dict[vid] = watch30

                    if f.startswith('25'):
                        news_view_rank_dict[vid] = view30
                        news_watch_rank_dict[vid] = watch30
            print('>>> Loading data: {0} done!'.format(os.path.join(subdir, f)))

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    plot_intersection(ax1, view_rank_dict, watch_rank_dict, color='r', linestyle='-', label='ALL')
    plot_intersection(ax1, music_view_rank_dict, music_watch_rank_dict, color='k', linestyle='--', label='Music')
    plot_intersection(ax1, news_view_rank_dict, news_watch_rank_dict, color='k', linestyle=':', label='News')
    ax1.set_ylim([0, 1])
    ax1.set_xlabel('top $n$ videos', fontsize=18)
    ax1.set_ylabel('Intersection $\\alpha$', fontsize=18)
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.legend(loc='lower right', handlelength=1, frameon=False, fontsize=20)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1)
    plot_spearman(ax2, view_rank_dict, watch_rank_dict, color='r', linestyle='-', label='ALL')
    plot_spearman(ax2, music_view_rank_dict, music_watch_rank_dict, color='k', linestyle='--', label='Music')
    plot_spearman(ax2, news_view_rank_dict, news_watch_rank_dict, color='k', linestyle=':', label='News')
    ax2.set_ylim([-1, 1])
    ax2.set_xlabel('top $n$ videos', fontsize=18)
    ax2.set_ylabel("Spearman's $\\rho$", fontsize=18)
    ax2.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.legend(loc='lower right', handlelength=1, frameon=False, fontsize=20)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.show()
