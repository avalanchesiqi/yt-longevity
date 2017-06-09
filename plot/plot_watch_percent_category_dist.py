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
                avg_watch = float(video['insights']['avgWatch']) * 60
                watch_percent = avg_watch / duration
                wp_list.append(watch_percent)


if __name__ == '__main__':
    category_id = '25'
    data_loc = '../../data/byCategory/{0}.json'.format(category_id)
    # data_loc = '../../data/byCategory/'

    # output_loc = '../linux_figs/wp_movie/{0}_wp'.format(folder_dict[category_id])

    # # make output dir if not exists, otherwise skip the program
    # if os.path.exists(output_loc):
    #     print 'output directory already exists! change output dir...'
    #     sys.exit(1)
    # else:
    #     os.makedirs(output_loc)

    wp_list = []
    if os.path.isdir(data_loc):
        for subdir, _, files in os.walk(data_loc):
            for f in files:
                filepath = os.path.join(subdir, f)
                update_matrix(filepath)
    else:
        update_matrix(data_loc)

    print 'done update matrix, now plot the figure...'

    fig, ax1 = plt.subplots(1, 1)
    ax1.hist(wp_list, bins=2500)

    ax1.set_xlim(xmin=0)
    ax1.set_xlim(xmax=1)
    ax1.set_xlabel('watch percentage')
    ax1.set_ylabel('frequency')
    ax1.set_title(category_dict[category_id])
    # fig.savefig(os.path.join(output_loc, 'week{0}.png'.format(week_idx + 1)), format='eps', dpi=1000)

    # plt.legend(loc='best')
    plt.show()
