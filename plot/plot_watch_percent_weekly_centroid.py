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

                # get seven days data for one figure
                for week_idx in np.arange(week_num):
                    valid_idx = (ages >= 7*week_idx) * (ages < 7*(week_idx+1))
                    weekly_view = np.sum(daily_view[valid_idx])
                    if weekly_view > 0:
                        weekly_watch = np.sum(daily_watch[valid_idx])
                        watch_percent = weekly_watch * 60 / weekly_view / duration

                        if watch_percent <= 1:
                            weekly_matrix[week_idx].append(watch_percent)
                        else:
                            weekly_matrix[week_idx].append(1)


if __name__ == '__main__':
    category_id = '43'
    data_loc = '../../data/byCategory/{0}.json'.format(category_id)
    # data_loc = '../../data/byCategory/'

    # output_loc = '../linux_figs/wp_movie/{0}_wp'.format(folder_dict[category_id])

    # # make output dir if not exists, otherwise skip the program
    # if os.path.exists(output_loc):
    #     print 'output directory already exists! change output dir...'
    #     sys.exit(1)
    # else:
    #     os.makedirs(output_loc)

    week_num = 52

    # get weekly statistics
    weekly_matrix = [[] for i in np.arange(week_num)]
    if os.path.isdir(data_loc):
        for subdir, _, files in os.walk(data_loc):
            for f in files:
                filepath = os.path.join(subdir, f)
                update_matrix(filepath)
    else:
        update_matrix(data_loc)

    print 'done update matrix, now plot the figure...'

    fig, ax1 = plt.subplots(1, 1)
    mean_weekly_statistics = [np.mean(x) for x in weekly_matrix]
    median_weekly_statistics = [np.median(x) for x in weekly_matrix]
    ax1.plot(np.arange(1, week_num+1), mean_weekly_statistics, 'r-', marker='o', label='Mean')
    ax1.plot(np.arange(1, week_num+1), median_weekly_statistics, 'b-', marker='s', label='Median')

    ax1.set_ylim(ymin=0)
    ax1.set_ylim(ymax=1)
    ax1.set_xlabel('week')
    ax1.set_ylabel('watch percentage')
    # fig.savefig(os.path.join(output_loc, 'week{0}.png'.format(week_idx + 1)), format='eps', dpi=1000)

    plt.legend(loc='best')
    plt.show()
