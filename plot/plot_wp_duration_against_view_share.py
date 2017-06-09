#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import numpy as np
from scipy import stats
import isodate
from datetime import datetime
from collections import defaultdict
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

category_dict = {"42": "Shorts", "29": "Nonprofits & Activism", "24": "Entertainment", "25": "News & Politics", "26": "Howto & Style", "27": "Education", "20": "Gaming", "21": "Videoblogging", "22": "People & Blogs", "23": "Comedy", "44": "Trailers", "28": "Science & Technology", "43": "Shows", "40": "Sci-Fi/Fantasy", "41": "Thriller", "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music", "39": "Horror", "38": "Foreign", "15": "Pets & Animals", "17": "Sports", "19": "Travel & Events", "18": "Short Movies", "31": "Anime/Animation", "30": "Movies", "37": "Family", "36": "Drama", "35": "Documentary", "34": "Comedy", "33": "Classics", "32": "Action/Adventure"}
folder_dict = {"42": "Shorts", "29": "Nonprofits", "24": "Entertainment", "25": "News", "26": "Howto", "27": "Education", "20": "Gaming", "21": "Videoblogging", "22": "People", "23": "Comedy", "44": "Trailers", "28": "Science", "43": "Shows", "40": "SciFi", "41": "Thriller", "1": "Film", "2": "Autos", "10": "Music", "39": "Horror", "38": "Foreign", "15": "Pets", "17": "Sports", "19": "Travel", "18": "ShortMovies", "31": "Anime", "30": "Movies", "37": "Family", "36": "Drama", "35": "Documentary", "34": "Comedy", "33": "Classics", "32": "Action"}


def read_as_int_array(content):
    return np.array(map(int, content.split(',')), dtype=np.uint32)


def read_as_float_array(content):
    return np.array(map(float, content.split(',')), dtype=np.float64)


def save_five_number_summary(quadruple):
    # boxplot
    fig, ax1 = plt.subplots(1, 1)
    bin_matrix = [[] for _ in xrange(bin_num)]
    for x in quadruple:
        bin_matrix[get_bin_idx(duration_list, x[0], bin_width, bin_num)].append(x[2]*60/x[1])
    print 'finish embedding into the matrix'
    ax1.boxplot(bin_matrix, showmeans=True, showfliers=False)

    # xticklabel
    xticklabels = []
    for i in xrange(bin_num):
        xticklabels.append('{0}%'.format(bin_width + bin_width * i))
    ax1.set_xticklabels(xticklabels)
    for label in ax1.get_xaxis().get_ticklabels():
        label.set_visible(False)
    for label in ax1.get_xaxis().get_ticklabels()[3::4]:
        label.set_visible(True)

    ax1.set_yscale('log')
    ax1.set_ylim(ymin=0)
    # ax1.set_ylim(ymax=1)
    # ax1.set_title('{0} at week 1'.format(title_text))
    ax1.set_xlabel('duration percentile')
    ax1.set_ylabel('first 8wk avg watch time (sec)')
    fig.savefig(os.path.join(output_loc, 'news_dur_vs_watch'))
    plt.show()


def get_bin_idx(overall_list, item, bin_width, bin_num):
    return min(int(stats.percentileofscore(overall_list, item)/bin_width), bin_num-1)


def str_to_date(time_string):
    return datetime(*map(int, time_string.split('-')))


def update_quadruple(filepath):
    with open(filepath, 'r') as filedata:
        for line in filedata:
            video = json.loads(line.rstrip())
            if video['insights']['dailyWatch'] == 'N' or video['insights']['dailyShare'] == 'N':
                continue
            duration = isodate.parse_duration(video['contentDetails']['duration']).seconds
            published_at = video['snippet']['publishedAt'][:10]
            if published_at[:4] == '2016' and duration > 0:
                start_date = video['insights']['startDate']
                time_diff = (str_to_date(start_date) - str_to_date(published_at)).days
                ages = read_as_int_array(video['insights']['days'])[:7*week_num] + time_diff
                daily_view = read_as_int_array(video['insights']['dailyView'])[:7*week_num]
                daily_watch = read_as_float_array(video['insights']['dailyWatch'])[:7*week_num]
                daily_share = read_as_int_array(video['insights']['dailyShare'])[:7*week_num]

                # get index for first week
                valid_idx = (ages >= 0) * (ages < 7*week_num)
                n_weeks_views = np.sum(daily_view[valid_idx])
                n_weeks_watches = np.sum(daily_watch[valid_idx])
                n_weeks_shares = np.sum(daily_share[valid_idx])
                if n_weeks_views > 0:
                    n_week_watch_time = n_weeks_watches*60/n_weeks_views

                    if 0 < n_week_watch_time <= duration:
                        vid = video['id']
                        quadruple_dict[vid].append(duration)
                        quadruple_dict[vid].append(n_weeks_views)
                        quadruple_dict[vid].append(n_weeks_watches)
                        quadruple_dict[vid].append(n_weeks_shares)


if __name__ == '__main__':
    category_id = '25'
    data_loc = '../../data/byCategory/{0}.json'.format(category_id)
    # data_loc = '../../data/byCategory'

    output_loc = '../linux_figs/'

    # 2.5 percentile or 5 percentile
    bin_width = 2.5
    bin_num = int(100/bin_width)
    week_num = 8

    # create a quadruple dict
    # video_id: duration, n_week_views, n_week_watches, n_week_shares
    quadruple_dict = defaultdict(list)

    if os.path.isdir(data_loc):
        for subdir, _, files in os.walk(data_loc):
            for f in files:
                filepath = os.path.join(subdir, f)
                update_quadruple(filepath)
    else:
        update_quadruple(data_loc)

    duration_list = [x[0] for x in quadruple_dict.values()]

    print 'done update matrix, now plot and save the figure...'

    save_five_number_summary(quadruple_dict.values())
