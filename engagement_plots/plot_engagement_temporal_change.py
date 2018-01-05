#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Scripts to plot relative engagement change over different ages, 30 days vs 60, 90, 120 days.

Usage: python plot_engagement_temporal_change.py
Time: ~30M
"""

from __future__ import print_function, division
import os, sys, time, datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
import pandas as pd
import cPickle as pickle
import matplotlib.pyplot as plt

from utils.helper import read_as_float_array, read_as_int_array
from utils.converter import to_relative_engagement


def _load_data(filepath):
    with open(filepath, 'r') as fin:
        fin.readline()
        for line in fin:
            _, _, duration, dump = line.split('\t', 3)
            _, days, views, watches = dump.rstrip().rsplit('\t', 3)

            duration = int(duration)
            days = read_as_int_array(days, delimiter=',', truncated=age)
            daily_view = read_as_int_array(views, delimiter=',', truncated=age)
            daily_watch = read_as_float_array(watches, delimiter=',', truncated=age)

            if np.sum(daily_view[days < 30]) == 0:
                continue

            for idx, t in enumerate([30, 60, 90, 120]):
                wp_t = np.sum(daily_watch[days < t]) * 60 / np.sum(daily_view[days < t]) / duration
                relative_engagement_quad[idx].append(to_relative_engagement(engagement_map, duration, wp_t, lookup_keys=lookup_durations))


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    print('>>> Start to plot temporal relative engagement change...')
    start_time = time.time()
    age = 120

    engagement_map_loc = '../data_preprocess/engagement_map.p'
    if not os.path.exists(engagement_map_loc):
        print('Engagement map not generated, start with generating engagement map first in ../data_preprocess dir!.')
        print('Exit program...')
        sys.exit(1)

    engagement_map = pickle.load(open(engagement_map_loc, 'rb'))
    lookup_durations = np.array(engagement_map['duration'])

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    output_path = 'temporal_engagement.csv'
    if not os.path.exists(output_path):
        input_loc = '../../production_data/new_tweeted_dataset_norm'
        relative_engagement_quad = [[], [], [], []]

        for subdir, _, files in os.walk(input_loc):
            for f in files:
                _load_data(os.path.join(subdir, f))
                print('>>> Loading data: {0} done!'.format(os.path.join(subdir, f)))
        print('>>> Finish loading all data!')
        re_frame = pd.DataFrame({'re30': relative_engagement_quad[0], 're60': relative_engagement_quad[1],
                                 're90': relative_engagement_quad[2], 're120': relative_engagement_quad[3]})
        re_frame.to_csv(output_path, sep='\t')
    else:
        re_frame = pd.read_csv(output_path, sep='\t')

    to_plot = True
    if to_plot:
        fig = plt.figure(figsize=(9, 9))
        ax1 = fig.add_subplot(111)

        re_change60 = []
        re_change90 = []
        re_change120 = []
        m = re_frame.shape[0]
        print('>>> Number of videos in dataset: ', m)
        # split into 60 bins, change less than 0.1, 50th bin, change less than -0.1, 30th bin
        start = -0.40
        end = 0.21
        step = 0.01
        x_axis = np.arange(start, end, step)
        for t in np.arange(start, end, step):
            re_change60.append(np.count_nonzero(re_frame['re60']-re_frame['re30'] <= t) / m)
            re_change90.append(np.count_nonzero(re_frame['re90']-re_frame['re30'] <= t) / m)
            re_change120.append(np.count_nonzero(re_frame['re120']-re_frame['re30'] <= t) / m)

        ax1.plot(x_axis, re_change60, 'r-', label=r'$|\bar \eta_{60}-\bar \eta_{30}|<0.1:' + '{0:.0f}\%$'.format(100*(re_change60[50]-re_change60[30])), lw=2)
        ax1.plot(x_axis, re_change90, 'g-', label=r'$|\bar \eta_{90}-\bar \eta_{30}|<0.1:' + '{0:.0f}\%$'.format(100*(re_change90[50]-re_change90[30])), lw=2)
        ax1.plot(x_axis, re_change120, 'b-', label=r'$|\bar \eta_{120}-\bar \eta_{30}|<0.1:' + '{0:.0f}\%$'.format(100*(re_change120[50]-re_change120[30])), lw=2)

        ax1.set_xlabel(r'relative engagement change $\Delta \bar \eta$', fontsize=24)
        ax1.set_ylabel('CDF', fontsize=24)
        ax1.set_xlim([-0.4, 0.2])
        ax1.set_ylim([0, 1])
        ax1.tick_params(axis='both', which='major', labelsize=24)
        ax1.legend(loc='upper left', fontsize=24, frameon=False, handletextpad=0.2, borderaxespad=0.5, handlelength=1, handleheight=1)
        ax1.set_xticks([-0.4, -0.2, 0, 0.2])

        # get running time
        print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])

        plt.title('(b)', fontsize=24)
        plt.tight_layout()
        plt.show()
