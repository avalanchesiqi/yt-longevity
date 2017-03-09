#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import math
from scipy import stats
import isodate
from datetime import datetime
import matplotlib.pyplot as plt


def read_as_int_array(content):
    return np.array(map(int, content.split(',')))


def read_as_float_array(content):
    return np.array(map(float, content.split(',')))


def safe_div(a, b):
    """ ignore / 0, safe_div( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~ np.isfinite(c)] = 0  # -inf inf NaN
    return c


def plot_errorbar(watch_list, color='b', label_text=None):
    z_critical = stats.norm.ppf(q=0.95)

    mean_list = []
    error_list = []

    for watches in watch_list:
        if len(watches) == 0:
            mean_list.append(0)
            error_list.append(0)
        else:
            mean = np.mean(watches)
            std = np.std(watches)
            error = z_critical * (std / math.sqrt(len(watches)))
            mean_list.append(mean)
            error_list.append(error)

    ax1.errorbar(np.arange(len(watch_list)), mean_list, yerr=error_list, c=color, fmt='o-', markersize='2', label=label_text)
    ax1.set_ylim(ymin=0)
    ax1.set_ylim(ymax=1)
    ax1.set_xlabel('video age')
    ax1.set_ylabel('watch percentage')


if __name__ == '__main__':
    data_loc = '../../data/medium_most_tweeted'
    fig, ax1 = plt.subplots(1, 1)

    cnt0 = 0
    cnt1 = 0
    cnt2 = 0

    matrix = [[] for _ in np.arange(1800)]
    for subdir, _, files in os.walk(data_loc):
        for f in files:
            filepath = os.path.join(subdir, f)
            with open(filepath, 'r') as filedata:
                for line in filedata:
                    video = json.loads(line.rstrip())
                    if video['insights']['dailyWatch'] == 'N':
                        cnt2 += 1
                        continue
                    duration = isodate.parse_duration(video['contentDetails']['duration']).seconds
                    published_at = video['snippet']['publishedAt'][:10]
                    start_date = video['insights']['startDate']
                    time_diff = (datetime(*map(int, start_date.split('-'))) - datetime(*map(int, published_at.split('-')))).days
                    days = read_as_int_array(video['insights']['days']) + time_diff
                    views = read_as_int_array(video['insights']['dailyView'])
                    watches = read_as_float_array(video['insights']['dailyWatch'])
                    watch_percent = safe_div(watches*60, views*duration)
                    for idx, day in enumerate(days):
                        if day < 1800:
                            if watch_percent[idx] <= 1:
                                matrix[day].append(watch_percent[idx])
                                cnt0 += 1
                            else:
                                matrix[day].append(1)
                                cnt1 += 1

    print 'success: {0}, invalid: {1}, unavailable: {2}'.format(cnt0, cnt1, cnt2)

    valid_matrix = [x for x in matrix if x]
    plot_errorbar(valid_matrix, color='r', label_text='Overall')

    plt.legend(loc='upper right')
    plt.show()
