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


def update_matrix(filepath):
    with open(filepath, 'r') as filedata:
        for line in filedata:
            video = json.loads(line.rstrip())
            if video['insights']['dailyWatch'] == 'N':
                # cnt2 += 1
                continue
            duration = isodate.parse_duration(video['contentDetails']['duration']).seconds
            published_at = video['snippet']['publishedAt'][:10]
            start_date = video['insights']['startDate']
            if published_at[:4] == '2016':
                time_diff = (
                datetime(*map(int, start_date.split('-'))) - datetime(*map(int, published_at.split('-')))).days
                days = read_as_int_array(video['insights']['days']) + time_diff
                views = read_as_int_array(video['insights']['dailyView'])
                watches = read_as_float_array(video['insights']['dailyWatch'])
                watch_percent = safe_div(watches * 60, views * duration)
                for idx, day in enumerate(days):
                    if day < 1800:
                        if watch_percent[idx] <= 1:
                            matrix1[day].append(watch_percent[idx])
                            # cnt0 += 1
                        else:
                            matrix1[day].append(1)
                            # cnt1 += 1
            else:
                time_diff = (
                datetime(*map(int, start_date.split('-'))) - datetime(*map(int, published_at.split('-')))).days
                days = read_as_int_array(video['insights']['days']) + time_diff
                views = read_as_int_array(video['insights']['dailyView'])
                watches = read_as_float_array(video['insights']['dailyWatch'])
                watch_percent = safe_div(watches * 60, views * duration)
                for idx, day in enumerate(days):
                    if day < 1800:
                        if watch_percent[idx] <= 1:
                            matrix2[day].append(watch_percent[idx])
                            # cnt0 += 1
                        else:
                            matrix2[day].append(1)
                            # cnt1 += 1


if __name__ == '__main__':
    category_id = '25'
    data_loc = '../../data/byCategory/{0}.json'.format(category_id)
    fig, ax1 = plt.subplots(1, 1)

    # cnt0 = 0
    # cnt1 = 0
    # cnt2 = 0

    matrix1 = [[] for _ in np.arange(1800)]
    matrix2 = [[] for _ in np.arange(1800)]

    if os.path.isdir(data_loc):
        for subdir, _, files in os.walk(data_loc):
            for f in files:
                filepath = os.path.join(subdir, f)
                update_matrix(filepath)
    else:
        update_matrix(data_loc)

    # print 'success: {0}, invalid: {1}, unavailable: {2}'.format(cnt0, cnt1, cnt2)

    y2016_matrix = [x for x in matrix1 if x]
    non_y2016_matrix = [x for x in matrix2 if x]
    plot_errorbar(y2016_matrix, color='r', label_text='{0} 2016'.format(category_dict[category_id]))
    plot_errorbar(non_y2016_matrix, color='b', label_text='{0} non-2016'.format(category_dict[category_id]))

    plt.legend(loc='upper right')
    plt.show()
