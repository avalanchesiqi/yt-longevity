#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import isodate
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def read_as_int_array(content):
    return np.array(map(int, content.split(',')), dtype=np.uint32)


def read_as_float_array(content):
    return np.array(map(float, content.split(',')), dtype=np.float64)


def str_to_date(time_string):
    return datetime(*map(int, time_string.split('-')))


def get_watch_percentage_from_file(filepath):
    ret = []
    with open(filepath, 'r') as f:
        for line in f:
            video = json.loads(line.rstrip())
            if video is not None and 'insights' in video:
                if video['insights']['dailyWatch'] == 'N':
                    continue
                duration = isodate.parse_duration(video['contentDetails']['duration']).seconds
                published_at = video['snippet']['publishedAt'][:10]
                # if duration > 0 and published_at[:4] == '2016':
                if duration > 0:
                    start_date = video['insights']['startDate']
                    time_diff = (str_to_date(start_date) - str_to_date(published_at)).days
                    ages = read_as_int_array(video['insights']['days']) + time_diff
                    daily_view = read_as_int_array(video['insights']['dailyView'])
                    daily_watch = read_as_float_array(video['insights']['dailyWatch'])

                    weekly_view = []
                    weekly_watch = []
                    # get seven days data
                    for week_idx in np.arange(week_num):
                        valid_idx = (ages >= 7 * week_idx) * (ages < 7 * (week_idx + 1))
                        weekly_view.append(np.sum(daily_view[valid_idx]))
                        weekly_watch.append(np.sum(daily_watch[valid_idx]))

                    # get the watch percentage of first _week_num_
                    if np.sum(weekly_view[:week_num]) > 0:
                        watch_percentage = np.sum(weekly_watch[:week_num]*60/np.sum(weekly_view[:week_num])/duration)
                        if watch_percentage > 1:
                            watch_percentage = 1
                        ret.append(watch_percentage)
    return ret


def get_watch_percentage(filepath):
    wp_list = []
    if os.path.isdir(filepath):
        for subdir, _, files in os.walk(filepath):
            for f in files:
                wp_list.extend(get_watch_percentage_from_file(os.path.join(subdir, f)))
    else:
        wp_list.extend(get_watch_percentage_from_file(filepath))
    return np.array(wp_list)


if __name__ == '__main__':
    music_loc = '../../data/byCategory/10.json'
    vevo_loc = '../../data/vevo_entire_data'
    billboard_loc = '../output/billboard_2016.txt'

    week_num = 8

    music_wp_list = get_watch_percentage(music_loc)
    vevo_wp_list = get_watch_percentage(vevo_loc)
    billboard_wp_list = get_watch_percentage(billboard_loc)

    print '2016 music videos:', len(music_wp_list)
    print '2016 vevo videos:', len(vevo_wp_list)
    print '2016 billboard videos:', len(billboard_wp_list)

    sns.set_style('white')
    sns.kdeplot(billboard_wp_list, bw=0.05, label='Billboard')
    sns.kdeplot(music_wp_list, bw=0.05, label='Music')
    sns.kdeplot(vevo_wp_list, bw=0.05, label='VEVO')

    plt.xlim([0, 1])
    plt.xlabel('watch percentage')
    plt.ylabel('probability density')
    plt.legend(loc='upper left')

    plt.show()
