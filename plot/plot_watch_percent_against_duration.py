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


def plot_errorbar(watch_list, color='b', label_text=None):
    z_critical = stats.norm.ppf(q=0.95)
    sample_size = len(watch_list[0])

    mean_list = []
    error_list = []

    for watches in watch_list:
        if len(watches) == 0:
            mean_list.append(0)
            error_list.append(0)
        else:
            mean = np.mean(watches)
            std = np.std(watches)
            error = z_critical * (std / math.sqrt(sample_size))
            mean_list.append(mean)
            error_list.append(error)

    ax1.errorbar(np.arange(1200), mean_list, yerr=error_list, c=color, fmt='o-', markersize='2', label=label_text)
    ax1.set_ylim(ymin=0)
    ax1.set_ylim(ymax=1)
    ax1.set_xlabel('video duration (*10s)')
    ax1.set_ylabel('watch percentage')


if __name__ == '__main__':
    data_loc = '../../data/medium_most_tweeted'
    fig, ax1 = plt.subplots(1, 1)

    cnt0 = 0
    cnt1 = 0
    cnt2 = 0

    x_axis = []
    y_axis = []
    matrix = [[] for _ in np.arange(1200)]
    for subdir, _, files in os.walk(data_loc):
        for f in files:
            filepath = os.path.join(subdir, f)
            with open(filepath, 'r') as filedata:
                for line in filedata:
                    video = json.loads(line.rstrip())
                    if video['insights']['avgWatch'] == 'N':
                        cnt2 += 1
                        continue
                    duration = isodate.parse_duration(video['contentDetails']['duration']).seconds
                    avg_watch = float(video['insights']['avgWatch'])*60
                    if duration > 0:
                        watch_percent = avg_watch/duration
                        if avg_watch <= duration < 12000:
                            matrix[duration/10].append(watch_percent)
                            cnt0 += 1
                        else:
                            cnt1 += 1

    print 'success: {0}, invalid: {1}, unavailable: {2}'.format(cnt0, cnt1, cnt2)

    plot_errorbar(matrix, 'y')

    plt.show()
