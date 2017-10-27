#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters

    # == == == == == == == == Part 2: Load dataset == == == == == == == ==
    # Positive example: hidden gem
    quality_channel = {}
    with open('quality_channel.txt', 'r') as fin:
        channel_id = fin.readline()
        channel_name = fin.readline().rstrip()
        duration_list = []
        wp_list = []
        es_list = []
        for line in fin:
            duration, wp30, es30 = line.rstrip().split()
            duration_list.append(int(duration))
            wp_list.append(float(wp30))
            es_list.append(float(es30))
        quality_channel['title'] = channel_name
        quality_channel['duration'] = duration_list
        quality_channel['wp30'] = wp_list
        quality_channel['es30'] = es_list

    # Negative example: junk channel
    junk_channel = {}
    with open('junk_channel.txt', 'r') as fin:
        channel_id = fin.readline()
        channel_name = fin.readline().rstrip()
        duration_list = []
        wp_list = []
        es_list = []
        for line in fin:
            duration, wp30, es30 = line.rstrip().split()
            duration_list.append(int(duration))
            wp_list.append(float(wp30))
            es_list.append(float(es30))
        junk_channel['title'] = channel_name
        junk_channel['duration'] = duration_list
        junk_channel['wp30'] = wp_list
        junk_channel['es30'] = es_list

    print(np.mean(quality_channel['wp30']))
    print(np.mean(junk_channel['wp30']))
    print(np.mean(quality_channel['es30']))
    print(np.mean(junk_channel['es30']))

    # Plot examples
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    ax.plot((0, 1), (0, 1), 'k--', zorder=1)

    plot_ax1 = True
    if plot_ax1:
        ax.scatter(quality_channel['wp30'], quality_channel['es30'], facecolors='none', edgecolors='r', marker='o', lw=1.5, s=60*np.log10(quality_channel['duration']), label=quality_channel['title'])

    plot_ax2 = True
    if plot_ax2:
        ax.scatter(junk_channel['wp30'], junk_channel['es30'], facecolors='none', edgecolors='b', marker='^', lw=1.5, s=60*np.log10(junk_channel['duration']), label=junk_channel['title'])

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('watch percentage '+r'$\bar \mu_{30}$', fontsize=24)
    ax.set_ylabel('relative engagement '+r'$\bar \eta_{30}$', fontsize=24)
    ax.legend(loc='lower left', scatterpoints=2, frameon=True, fontsize=23)
    ax.tick_params(axis='both', which='major', labelsize=24)

    plt.tight_layout()
    plt.show()
