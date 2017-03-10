#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from collections import defaultdict
import numpy as np
from scipy.interpolate import Rbf
import isodate
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

category_dict = {"42": "Shorts", "29": "Nonprofits & Activism", "24": "Entertainment", "25": "News & Politics", "26": "Howto & Style", "27": "Education", "20": "Gaming", "21": "Videoblogging", "22": "People & Blogs", "23": "Comedy", "44": "Trailers", "28": "Science & Technology", "43": "Shows", "40": "Sci-Fi/Fantasy", "41": "Thriller", "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music", "39": "Horror", "38": "Foreign", "15": "Pets & Animals", "17": "Sports", "19": "Travel & Events", "18": "Short Movies", "31": "Anime/Animation", "30": "Movies", "37": "Family", "36": "Drama", "35": "Documentary", "34": "Comedy", "33": "Classics", "32": "Action/Adventure"}


def x_fmt(x, p):
    return int(10*x)


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


if __name__ == '__main__':
    category_id = '17'
    data_loc = '../../data/byCategory/{0}.json'.format(category_id)

    age_duration_matrix = defaultdict(dict)
    with open(data_loc, 'r') as filedata:
        for line in filedata:
            video = json.loads(line.rstrip())
            if video['insights']['dailyWatch'] == 'N':
                continue
            duration = isodate.parse_duration(video['contentDetails']['duration']).seconds
            # for videos shorter than 1 hour
            if duration < 3600:
                published_at = video['snippet']['publishedAt'][:10]
                if published_at[:4] == '2016':
                    start_date = video['insights']['startDate']
                    time_diff = (datetime(*map(int, start_date.split('-'))) - datetime(*map(int, published_at.split('-')))).days
                    days = read_as_int_array(video['insights']['days']) + time_diff
                    views = read_as_int_array(video['insights']['dailyView'])
                    watches = read_as_float_array(video['insights']['dailyWatch'])
                    watch_percent = safe_div(watches * 60, views * duration)
                    for idx, age in enumerate(days):
                        # for videos before age 400 days
                        if 0 <= age < 400:
                            # age idx
                            age_idx = age/10
                            # duration idx
                            duration_idx = duration/60
                            if watch_percent[idx] <= 1:
                                wp_value = watch_percent[idx]
                            else:
                                wp_value = 1
                            if duration_idx not in age_duration_matrix[age_idx].keys():
                                age_duration_matrix[age_idx][duration_idx] = []
                            age_duration_matrix[age_idx][duration_idx].append(wp_value)

    age_duration_mean_matrix = defaultdict(dict)
    for age_idx in age_duration_matrix.keys():
        for duration_idx in age_duration_matrix[age_idx].keys():
            age_duration_mean_matrix[age_idx][duration_idx] = np.mean(age_duration_matrix[age_idx][duration_idx])

    # change to list presentation
    age_list = []
    duration_list = []
    watch_percent_list = []
    for age_idx in age_duration_mean_matrix.keys():
        for duration_idx in age_duration_mean_matrix[age_idx].keys():
            age_list.append(age_idx)
            duration_list.append(duration_idx)
            watch_percent_list.append(age_duration_mean_matrix[age_idx][duration_idx])

    # Creating the output grid (60x60, in the example)
    ti = np.linspace(0, 60.0, 60)
    xi, yi = np.meshgrid(ti, ti)

    # Creating the interpolation function and populating the output matrix value
    rbf = Rbf(age_list, duration_list, watch_percent_list, function='inverse')
    zi = rbf(xi, yi)

    # Plotting the result
    plt.subplot(1, 1, 1)
    plt.pcolor(xi, yi, zi, vmin=0, vmax=1)
    # plt.scatter(age_list, duration_list, c=watch_percent_list)
    plt.title('heatmap of category {0}'.format(category_dict[category_id]))
    plt.xlim(0, 40)
    plt.ylim(0, 60)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(x_fmt))
    plt.xlabel('age (day)')
    plt.ylabel('duration (minute)')
    plt.colorbar()

    plt.show()

