#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def get_percentile(duration, true_wp):
    bin_idx = np.sum(duration_split_points < duration)
    duration_bin = dur_engage_map[bin_idx]
    wp_percentile = np.sum(np.array(duration_bin) < true_wp) / 1000
    return wp_percentile


def get_percentile_list(duration_list, wp_list):
    percentile_list = []
    for d, p in zip(duration_list, wp_list):
        percentile_list.append(get_percentile(d, p))
    return percentile_list


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    dur_engage_str_map = pickle.load(open('dur_engage_map.p', 'rb'))
    dur_engage_map = {key: list(map(float, value.split(','))) for key, value in dur_engage_str_map.items()}

    duration_split_points = np.array(dur_engage_map['duration'])

    # == == == == == == == == Part 2: Load dataset == == == == == == == ==
    # Positive example: hidden gem
    pos_example = {'title': 'Previously Recorded', 'duration': [612, 7529, 8021, 12694, 4097, 6404, 1616, 6041],
                   'wp': [0.619681378034, 0.256171746859, 0.2207092558, 0.0914135185341, 0.316452704991, 0.313239578608, 0.552658131973, 0.242933920311]}

    # Negative example: junk channel
    neg_example = {'title': 'Voice Actors Play', 'duration': [181, 1137, 30, 717, 10530, 79, 3618, 27237, 18064],
                   'wp': [0.405596083826, 0.0725868513633, 0.871891891893, 0.108607076308, 0.0193880131904, 0.576285884394, 0.0630317650676, 0.00563614976038, 0.0198870835794]}

    # Plot examples
    fig = plt.figure(figsize=(6, 6))

    plot_ax1 = False
    if plot_ax1:
        ax1 = fig.add_subplot(111)
        percentile_example1 = get_percentile_list(pos_example['duration'], pos_example['wp'])
        ax1.scatter(np.log10(pos_example['duration']), pos_example['wp'], c='b', lw=1, s=30, label='Original', zorder=10)
        ax1.scatter(np.log10(pos_example['duration']), percentile_example1, c='r', lw=1, s=30, label='Transformed', zorder=10)
        ax1.set_title(pos_example['title'])
        for i in xrange(len(pos_example['duration'])):
            ax1.arrow(np.log10(pos_example['duration'])[i], pos_example['wp'][i], 0, percentile_example1[i]-pos_example['wp'][i]-0.03, head_width=0.04, head_length=0.03,
                      alpha=0.5, ec='k')

    plot_ax2 = True
    if plot_ax2:
        ax2 = fig.add_subplot(111)
        percentile_example2 = get_percentile_list(neg_example['duration'], neg_example['wp'])
        ax2.scatter(np.log10(neg_example['duration']), neg_example['wp'], c='b', lw=1, s=30, label='Original', zorder=10)
        ax2.scatter(np.log10(neg_example['duration']), percentile_example2, c='r', lw=1, s=30, label='Transformed', zorder=10)
        ax2.set_title(neg_example['title'])
        for i in xrange(len(neg_example['duration'])):
            if percentile_example2[i]-neg_example['wp'][i] > 0:
                ax2.arrow(np.log10(neg_example['duration'])[i], neg_example['wp'][i], 0, percentile_example2[i]-neg_example['wp'][i]-0.03, head_width=0.04, head_length=0.03,
                          alpha=0.5, ec='k')
            else:
                ax2.arrow(np.log10(neg_example['duration'])[i], neg_example['wp'][i], 0,
                          percentile_example2[i] - neg_example['wp'][i] + 0.03, head_width=0.03, head_length=0.03,
                          alpha=0.5, ec='k')

    def exponent(x, pos):
        'The two args are the value and tick position'
        return '%1.0f' % (10**x)

    x_formatter = FuncFormatter(exponent)

    for ax in [ax2]:
        ax.xaxis.set_major_formatter(x_formatter)
        for label in ax.get_xticklabels()[1::2]:
            label.set_visible(False)

        ax.set_xlim([1, 5])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Video duration (sec)', fontsize=16)
        ax.set_ylabel('Watch percentage', fontsize=16)
        ax.legend(loc='lower left')

    plt.tight_layout()
    plt.show()
