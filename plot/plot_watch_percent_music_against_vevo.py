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
    return np.array(map(int, content.split(',')), dtype=np.uint32)


def read_as_float_array(content):
    return np.array(map(float, content.split(',')), dtype=np.float64)


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
    matrix = [[] for _ in np.arange(1800)]
    with open(filepath, 'r') as filedata:
        for line in filedata:
            video = json.loads(line.rstrip())
            if video['insights']['dailyWatch'] == 'N':
                continue
            duration = isodate.parse_duration(video['contentDetails']['duration']).seconds
            published_at = video['snippet']['publishedAt'][:10]
            start_date = video['insights']['startDate']
            if published_at[:4] == '2016':
                time_diff = (datetime(*map(int, start_date.split('-'))) - datetime(*map(int, published_at.split('-')))).days
                days = read_as_int_array(video['insights']['days']) + time_diff
                views = read_as_int_array(video['insights']['dailyView'])
                watches = read_as_float_array(video['insights']['dailyWatch'])
                watch_percent = safe_div(watches * 60, views * duration)
                for idx, day in enumerate(days):
                    if day < 1800:
                        if watch_percent[idx] <= 1:
                            matrix[day].append(watch_percent[idx])
                        else:
                            matrix[day].append(1)
    return matrix


def update_matrix_dir(filepath):
    matrix = [[] for _ in np.arange(1800)]
    for subdir, _, files in os.walk(filepath):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as filedata:
                for line in filedata:
                    video = json.loads(line.rstrip())
                    if video['insights']['dailyWatch'] == 'N':
                        continue
                    duration = isodate.parse_duration(video['contentDetails']['duration']).seconds
                    published_at = video['snippet']['publishedAt'][:10]
                    start_date = video['insights']['startDate']
                    if published_at[:4] == '2016':
                        time_diff = (datetime(*map(int, start_date.split('-'))) - datetime(*map(int, published_at.split('-')))).days
                        days = read_as_int_array(video['insights']['days']) + time_diff
                        views = read_as_int_array(video['insights']['dailyView'])
                        watches = read_as_float_array(video['insights']['dailyWatch'])
                        watch_percent = safe_div(watches * 60, views * duration)
                        for idx, day in enumerate(days):
                            if day < 1800:
                                if watch_percent[idx] <= 1:
                                    matrix[day].append(watch_percent[idx])
                                else:
                                    matrix[day].append(1)
    return matrix


if __name__ == '__main__':
    data_loc = '../../data/byCategory/10.json'
    vevo_loc = '../../data/vevo_entire_data'
    fig, ax1 = plt.subplots(1, 1)

    music_matrix = update_matrix(data_loc)
    vevo_matrix = update_matrix_dir(vevo_loc)

    music_matrix = [x for x in music_matrix if x]
    vevo_matrix = [x for x in vevo_matrix if x]
    plot_errorbar(music_matrix, color='b', label_text='Music 2016')
    plot_errorbar(vevo_matrix, color='r', label_text='VEVO 2016')

    plt.legend(loc='upper right')
    plt.show()
