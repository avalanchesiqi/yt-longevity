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

category_dict = {"42": "Shorts", "29": "Nonprofits & Activism", "24": "Entertainment", "25": "News & Politics", "26": "Howto & Style", "27": "Education", "20": "Gaming", "21": "Videoblogging", "22": "People & Blogs", "23": "Comedy", "44": "Trailers", "28": "Science & Technology", "43": "Shows", "40": "Sci-Fi/Fantasy", "41": "Thriller", "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music", "39": "Horror", "38": "Foreign", "15": "Pets & Animals", "17": "Sports", "19": "Travel & Events", "18": "Short Movies", "31": "Anime/Animation", "30": "Movies", "37": "Family", "36": "Drama", "35": "Documentary", "34": "Comedy", "33": "Classics", "32": "Action/Adventure"}


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

    ax1.errorbar(np.arange(1200), mean_list, yerr=error_list, c=color, fmt='o-', markersize='2', label=label_text)
    ax1.set_ylim(ymin=0)
    # ax1.set_ylim(ymax=1)
    ax1.set_xlim(xmin=0)
    ax1.set_xlim(xmax=1200)
    ax1.set_xlabel('video duration (*10s)')
    ax1.set_ylabel('average watch time (1s)')


def update_matrix(filepath):
    with open(filepath, 'r') as filedata:
        for line in filedata:
            video = json.loads(line.rstrip())
            if video['insights']['avgWatch'] == 'N':
                # cnt2 += 1
                continue
            duration = isodate.parse_duration(video['contentDetails']['duration']).seconds
            avg_watch = float(video['insights']['avgWatch']) * 60
            if duration > 0:
                watch_percent = avg_watch / duration
                if avg_watch <= duration < 12000:
                    matrix[duration / 10].append(watch_percent)
                    # cnt0 += 1
                # else:
                    # cnt1 += 1


if __name__ == '__main__':
    category_id = '25'
    data_loc = '../../data/byCategory/{0}.json'.format(category_id)
    fig, ax1 = plt.subplots(1, 1)

    # cnt0 = 0
    # cnt1 = 0
    # cnt2 = 0

    matrix = [[] for _ in np.arange(1200)]

    if os.path.isdir(data_loc):
        for subdir, _, files in os.walk(data_loc):
            for f in files:
                filepath = os.path.join(subdir, f)
                update_matrix(filepath)
    else:
        update_matrix(data_loc)

    # print 'success: {0}, invalid: {1}, unavailable: {2}'.format(cnt0, cnt1, cnt2)

    plot_errorbar(matrix, color='y', label_text='{0}'.format(category_dict[category_id]))

    plt.legend(loc='upper right')
    plt.show()
