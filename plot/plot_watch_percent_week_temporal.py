#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import numpy as np
from scipy import stats
import isodate
from datetime import datetime
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

category_dict = {"42": "Shorts", "29": "Nonprofits & Activism", "24": "Entertainment", "25": "News & Politics", "26": "Howto & Style", "27": "Education", "20": "Gaming", "21": "Videoblogging", "22": "People & Blogs", "23": "Comedy", "44": "Trailers", "28": "Science & Technology", "43": "Shows", "40": "Sci-Fi/Fantasy", "41": "Thriller", "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music", "39": "Horror", "38": "Foreign", "15": "Pets & Animals", "17": "Sports", "19": "Travel & Events", "18": "Short Movies", "31": "Anime/Animation", "30": "Movies", "37": "Family", "36": "Drama", "35": "Documentary", "34": "Comedy", "33": "Classics", "32": "Action/Adventure"}
folder_dict = {"42": "Shorts", "29": "Nonprofits", "24": "Entertainment", "25": "News", "26": "Howto", "27": "Education", "20": "Gaming", "21": "Videoblogging", "22": "People", "23": "Comedy", "44": "Trailers", "28": "Science", "43": "Shows", "40": "SciFi", "41": "Thriller", "1": "Film", "2": "Autos", "10": "Music", "39": "Horror", "38": "Foreign", "15": "Pets", "17": "Sports", "19": "Travel", "18": "ShortMovies", "31": "Anime", "30": "Movies", "37": "Family", "36": "Drama", "35": "Documentary", "34": "Comedy", "33": "Classics", "32": "Action"}


def read_as_int_array(content):
    return np.array(map(int, content.split(',')), dtype=np.uint32)


def read_as_float_array(content):
    return np.array(map(float, content.split(',')), dtype=np.float64)


def plot_temporal_week(week_matrix, title_text=None):
    fig, ax1 = plt.subplots(1, 1)

    week1 = np.array(week_matrix[0])
    week2 = np.array(week_matrix[1])

    print '{0} videos decrease watch percentage in category {1}'.format(np.sum(week1 > week2), title_text)
    print '{0} videos increase watch percentage in category {1}'.format(np.sum(week1 < week2), title_text)

    # decrease wp in week2
    ax1.scatter(week1[week1 > week2], week2[week1 > week2], c='b', s=5, lw=0, label='decrease watch')
    # increase wp in week2
    ax1.scatter(week1[week1 < week2], week2[week1 < week2], c='r', s=5, lw=0, label='increase watch')
    # equal wp in week2
    ax1.scatter(week1[week1 == week2], week2[week1 == week2], c='g', s=5, lw=0)

    ax1.set_xlim(xmin=0)
    ax1.set_xlim(xmax=1)
    ax1.set_ylim(ymin=0)
    ax1.set_ylim(ymax=1)
    ax1.set_title('Evolution of {0} videos'.format(title_text))
    ax1.set_xlabel('watch percentage at week1')
    ax1.set_ylabel('watch percentage at week2')
    # equal aspect
    ax1.set_aspect('equal')


def update_matrix(filepath):
    with open(filepath, 'r') as filedata:
        for line in filedata:
            video = json.loads(line.rstrip())
            if video['insights']['avgWatch'] == 'N':
                continue
            duration = isodate.parse_duration(video['contentDetails']['duration']).seconds
            published_at = video['snippet']['publishedAt'][:10]
            if published_at[:4] == '2016' and duration > 0:
                start_date = video['insights']['startDate']
                time_diff = (datetime(*map(int, start_date.split('-'))) - datetime(*map(int, published_at.split('-')))).days
                ages = read_as_int_array(video['insights']['days'])[:7*week_num] + time_diff
                daily_view = read_as_int_array(video['insights']['dailyView'])[:7*week_num]
                daily_watch = read_as_float_array(video['insights']['dailyWatch'])[:7*week_num]

                week_idx1 = 1
                week_idx2 = 2
                # get seven days data for one figure
                valid_idx1 = (ages >= 7*(week_idx1-1)) * (ages < 7*week_idx1)
                weekly_view1 = np.sum(daily_view[valid_idx1])
                valid_idx2 = (ages >= 7*(week_idx2-1)) * (ages < 7*week_idx2)
                weekly_view2 = np.sum(daily_view[valid_idx2])
                if weekly_view1 > 100 and weekly_view2 > 100:
                    weekly_watch1 = np.sum(daily_watch[valid_idx1])
                    watch_percent1 = weekly_watch1 * 60 / weekly_view1 / duration
                    weekly_watch2 = np.sum(daily_watch[valid_idx2])
                    watch_percent2 = weekly_watch2 * 60 / weekly_view2 / duration

                    if watch_percent1 <= 1:
                        week_matrix[0].append(watch_percent1)
                    else:
                        week_matrix[0].append(1)
                    if watch_percent2 <= 1:
                        week_matrix[1].append(watch_percent2)
                    else:
                        week_matrix[1].append(1)


if __name__ == '__main__':
    category_id = '43'
    data_loc = '../../data/byCategory/{0}.json'.format(category_id)

    output_loc = '../linux_figs/wp_movie/{0}_wp'.format(folder_dict[category_id])

    # make output dir if not exists, otherwise skip the program
    # if os.path.exists(output_loc):
    #     print 'output directory already exists! change output dir...'
    #     sys.exit(1)
    # else:
    #     os.makedirs(output_loc)

    # 2.5 percentile or 5 percentile
    bin_width = 2.5
    bin_num = int(100/bin_width)
    week_num = 2

    # get weekly bin statistics
    week_matrix = [[] for i in np.arange(week_num)]
    if os.path.isdir(data_loc):
        for subdir, _, files in os.walk(data_loc):
            for f in files:
                filepath = os.path.join(subdir, f)
                update_matrix(filepath)
    else:
        update_matrix(data_loc)

    print 'done updating matrix, now plot and save the figure...'

    plot_temporal_week(week_matrix, title_text='{0}'.format(category_dict[category_id]))

    # plt.legend(loc='upper left')
    plt.show()
