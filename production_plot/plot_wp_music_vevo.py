#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import sys
import os
import json
import numpy as np
from scipy import stats
import isodate
from datetime import datetime
import matplotlib.pyplot as plt


def normalized_bin(x1, x2, ax, text=None):
    hist1, bins1 = np.histogram(x1, bins=20)
    hist2, bins2 = np.histogram(x2, bins=20)
    sum_val1 = sum(hist1)
    sum_val2 = sum(hist2)
    hist1 = [float(n) / sum_val1 for n in hist1]
    hist2 = [float(n) / sum_val2 for n in hist2]

    # Plot the resulting histogram
    center1 = (bins1[:-1] + bins1[1:]) / 2
    width = 0.35 * (bins1[1] - bins1[0])
    ax.bar(center1, hist1, align='center', width=width, color='b', label='Random')
    ax.bar(center1+width, hist2, align='center', width=width, color='r', label='Quality')

    ax.set_xlim(xmin=0)
    ax.set_xlim(xmax=1)
    if text is not None:
        ax.set_xlabel(text)
    else:
        ax.set_xlabel('Watch Percentage')
    ax.set_ylabel('Fraction of Videos')
    ax.legend(loc='upper left', fontsize='small')


def save_five_number_summary(bin_matrix, bin_matrix2, bin_matrix3, bin_matrix4):
    # fig, ax1 = plt.subplots(1, 1)
    fig = plt.figure(figsize=(10, 10))

    ax1, ax3, ax5 = [fig.add_subplot(321 + 2*idx) for idx in xrange(3)]
    ax2, ax4, ax6 = [fig.add_subplot(322 + 2*idx) for idx in xrange(3)]

    # ax1.boxplot(bin_matrix, showmeans=False, showfliers=False)
    ax1.errorbar(np.arange(1, 1+len(bin_matrix)), [np.median(x) for x in bin_matrix], yerr=[[np.median(x)-np.percentile(x, 25) for x in bin_matrix], [np.percentile(x, 75)-np.median(x) for x in bin_matrix]], fmt='-ob', label='Random Music')
    ax1.errorbar(np.arange(1, 1+len(bin_matrix2)), [np.median(x) for x in bin_matrix2], yerr=[[np.median(x)-np.percentile(x, 25) for x in bin_matrix2], [np.percentile(x, 75)-np.median(x) for x in bin_matrix2]], fmt='-dr', label='VEVO')

    # for popu in np.arange(10, 100, 10):
    #     ax1.plot(np.arange(1, 1 + len(bin_matrix)), [np.percentile(x, popu) for x in bin_matrix], '-')
    #
    # ax1.plot(np.arange(1, 1 + len(bin_matrix2)), [np.median(x) for x in bin_matrix2], '-o', label='VEVO')

    # ax1.plot(np.arange(1, 1+len(bin_matrix2)), [np.median(x) for x in bin_matrix2], 'go--', label='VEVO')
    # ax2.boxplot(bin_matrix2, showmeans=False, showfliers=False)
    ax2.errorbar(np.arange(1, 1+len(bin_matrix3)), [np.median(x) for x in bin_matrix3], yerr=[[np.median(x)-np.percentile(x, 25) for x in bin_matrix3], [np.percentile(x, 75)-np.median(x) for x in bin_matrix3]], fmt='-ob', label='Random News')
    ax2.errorbar(np.arange(1, 1+len(bin_matrix4)), [np.median(x) for x in bin_matrix4], yerr=[[np.median(x)-np.percentile(x, 25) for x in bin_matrix4], [np.percentile(x, 75)-np.median(x) for x in bin_matrix4]], fmt='-dr', label='Top News')

    normalized_bin(bin_matrix[10], bin_matrix2[10], ax3, text='Watch Percentage @bin10, Music')
    normalized_bin(bin_matrix[15], bin_matrix2[15], ax4, text='Watch Percentage @bin15, Music')
    normalized_bin(bin_matrix3[10], bin_matrix4[10], ax5, text='Watch Percentage @bin10, News')
    normalized_bin(bin_matrix3[15], bin_matrix4[15], ax6, text='Watch Percentage @bin15, News')

    for ax in (ax1, ax2):
        # xticklabel
        xticklabels = []
        for i in xrange(bin_num):
            xticklabels.append('{0}%'.format(12.5 * i))
        ax.set_xticklabels(xticklabels)
        for label in ax.get_xaxis().get_ticklabels():
            label.set_visible(False)
        for label in ax.get_xaxis().get_ticklabels()[0::2]:
            label.set_visible(True)

        ax.set_xlim(xmin=1)
        ax.set_xlim(xmax=bin_num)
        ax.set_ylim(ymin=0)
        ax.set_ylim(ymax=1)
        ax.set_xlabel('video duration percentile')
        ax.set_ylabel('watch percentage')
        ax.legend(loc="upper right", fontsize='small')

    # ax1.set_title('Music and VEVO watch percentage by day 30')
    # ax2.set_title('Vevo watch percentage by day 30')

if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    age = 30
    view_percent_matrix = None
    cnt = 0
    bin_width = 2.5
    bin_num = int(100 / bin_width)

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    music_loc = '../../data/production_data/music_wp_dur.txt'
    vevo_loc = '../../data/production_data/vevo_wp_dur.txt'
    news_loc = '../../data/production_data/news_wp_dur.txt'
    top_news_loc = '../../data/production_data/top_news_wp_dur.txt'
    # vevo_loc = '../../data/production_data/vevo_wp_dur.txt'

    music_tuple = []
    with open(music_loc, 'r') as music_data:
        for line in music_data:
            duration, wp = line.rstrip().split()
            duration = int(duration)
            wp = float(wp)
            music_tuple.append(tuple((duration, wp)))

    news_tuple = []
    with open(news_loc, 'r') as news_data:
        for line in news_data:
            duration, wp = line.rstrip().split()
            duration = int(duration)
            wp = float(wp)
            news_tuple.append(tuple((duration, wp)))

    print 'finish loading'

    music_sorted_tuple = sorted(music_tuple, key=lambda x: x[0])
    boundary_list = []

    news_sorted_tuple = sorted(news_tuple, key=lambda x: x[0])
    boundary_list2 = []

    # get daily bin statistics
    music_bin_matrix = [[] for _ in np.arange(bin_num)]
    music_elem_num = len(music_sorted_tuple)//bin_num + 1
    music_idx = 0
    for i, v in enumerate(music_sorted_tuple):
        if i % music_elem_num == 0 and i != 0:
            music_idx += 1
            boundary_list.append(v[0])
        music_bin_matrix[music_idx].append(v[1])

    vevo_bin_matrix = [[] for _ in np.arange(bin_num)]
    with open(vevo_loc, 'r') as vevo_data:
        for line in vevo_data:
            duration, wp = line.rstrip().split()
            duration = int(duration)
            wp = float(wp)
            vevo_idx = 0
            for j, k in enumerate(boundary_list):
                if duration >= k:
                    vevo_idx = j+1
                else:
                    break
            vevo_bin_matrix[vevo_idx].append(wp)

    news_bin_matrix = [[] for _ in np.arange(bin_num)]
    news_elem_num = len(news_sorted_tuple) // bin_num + 1
    news_idx = 0
    for i, v in enumerate(news_sorted_tuple):
        if i % news_elem_num == 0 and i != 0:
            news_idx += 1
            boundary_list2.append(v[0])
        news_bin_matrix[news_idx].append(v[1])

    tnews_bin_matrix = [[] for _ in np.arange(bin_num)]
    with open(top_news_loc, 'r') as tnews_data:
        for line in tnews_data:
            duration, wp = line.rstrip().split()
            duration = int(duration)
            wp = float(wp)
            tnews_idx = 0
            for j, k in enumerate(boundary_list2):
                if duration >= k:
                    tnews_idx = j + 1
                else:
                    break
            tnews_bin_matrix[tnews_idx].append(wp)

    print 'prepare to plot'
    save_five_number_summary(music_bin_matrix, vevo_bin_matrix, news_bin_matrix, tnews_bin_matrix)

    plt.tight_layout()
    plt.show()
    print 'done update matrix, now plot and save the figure...'
