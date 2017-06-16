#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import sys
import os
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Plot theta from watch percentage temporal fit
# Usage: python plot_wp_fitted_theta.py /Volumes/mbp/Users/siqi/OData/fitted_log


def extract_value(content):
    """
    Read float in a 'xxx: float' format.
    :param content: string input
    :return: a float value
    """
    return float(content.rsplit(':', 1)[1])


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    fig, (ax1, ax2) = plt.subplots(2, 1)
    age = 180
    exp_rmse = []
    linear_rmse = []
    constant_rmse = []
    theta_array = []

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    data_loc = sys.argv[1]

    for subdir, _, files in os.walk(data_loc):
        for f in files:
            with open(os.path.join(subdir, f)) as fin:
                # video id
                # exponential rmse, linear rmse, constant rmse
                # mu, theta, weight, bias, const
                while True:
                    vid_line = fin.readline().rstrip()
                    if vid_line == '':
                        break
                    exp, linear, constant = fin.readline().rstrip().split(',')
                    mu, theta, weight, bias, const = fin.readline().rstrip().split(',')

                    exp_rmse.append(extract_value(exp))
                    linear_rmse.append(extract_value(linear))
                    constant_rmse.append(extract_value(constant))
                    theta_array.append(extract_value(theta))

    # == == == == == == == == Part 3: Plot dataset == == == == == == == == #
    ax1.boxplot([exp_rmse, linear_rmse, constant_rmse], labels=['exponential', 'linear', 'constant'], showfliers=False, showmeans=True)
    ax1.set_ylabel('RMSE')

    medians = [np.median(exp_rmse), np.median(linear_rmse), np.median(constant_rmse)]
    median_labels = [str(np.round(s, 4)) for s in medians]
    pos = range(len(medians))
    for tick, label in zip(pos, ax1.get_xticklabels()):
        ax1.text(pos[tick]+1, medians[tick]-0.005, median_labels[tick],
                 horizontalalignment='center', size='small', color='k')

    threshold = 0.1
    lower_bound = -np.log(1+threshold)/np.log(180)
    upper_bound = -np.log(1-threshold)/np.log(180)
    lowest_bound = np.percentile(theta_array, 0.03)
    highest_bound = np.percentile(theta_array, 99.7)
    print('lowest bound:', np.percentile(theta_array, 0.03))
    print('lower bound:', lower_bound)
    print('upper bound:', upper_bound)
    print('highest bound:', np.percentile(theta_array, 99.7))

    growth_videos = [x for x in theta_array if lowest_bound < x < lower_bound]
    memoryless_videos = [x for x in theta_array if lower_bound <= x <= upper_bound]
    decay_videos = [x for x in theta_array if upper_bound < x < highest_bound]
    ax2.hist(growth_videos, color='r', bins=int(round((lower_bound-lowest_bound)*100)))
    ax2.hist(memoryless_videos, color='g', bins=int(round((upper_bound-lower_bound)*100)))
    ax2.hist(decay_videos, color='b', bins=int(round((highest_bound-upper_bound)*100)))
    ax2.set_xlabel('theta')
    ax2.set_ylabel('frequency')

    growth_videos_ratio = len(growth_videos)/len(theta_array)*100
    memoryless_videos_ratio = len(memoryless_videos)/len(theta_array)*100
    decay_videos_ratio = len(decay_videos)/len(theta_array)*100

    print('growth videos: {0:.2f}%'.format(growth_videos_ratio))
    print('memoryless videos: {0:.2f}%'.format(memoryless_videos_ratio))
    print('decay videos: {0:.2f}%'.format(decay_videos_ratio))

    ax2.text(lowest_bound, 6000, 'threshold: {0}\ntotal videos: {1}\ngrowth videos: {2:.2f}%\n'
                                 'memoryless videos: {3:.2f}%\ndecay videos: {4:.2f}%'
             .format(threshold, len(theta_array), growth_videos_ratio, memoryless_videos_ratio, decay_videos_ratio),
             bbox={'facecolor': 'green', 'alpha': 0.5})

    # fig.savefig('wp_rmse_comp_and_theta_dist')
    plt.show()
