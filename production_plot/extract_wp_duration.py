#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import sys
import os
import json
import numpy as np
from scipy import stats
import isodate
from datetime import datetime


def read_as_int_array(content, truncated=None):
    if truncated is None:
        return np.array(map(int, content.split(',')), dtype=np.uint32)
    else:
        return np.array(map(int, content.split(',')), dtype=np.uint32)[:truncated]


def read_as_float_array(content, truncated=None):
    if truncated is None:
        return np.array(map(float, content.split(',')), dtype=np.float64)
    else:
        return np.array(map(float, content.split(',')), dtype=np.float64)[:truncated]


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    age = 30
    view_percent_matrix = None
    cnt = 0

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    category_id = '25'
    # data_loc = '../../data/production_data/random_dataset/{0}.json'.format(category_id)
    data_loc = '../../data/production_data/popular_news'

    fout = open('../../data/production_data/top_news_wp_dur.txt', 'w')

    # with open(data_loc, 'r') as fin:
    #     for line in fin:
    #         video = json.loads(line.rstrip())
    #         if video['insights']['dailyWatch'] == 'N':
    #             continue
    #         published_at = video['snippet']['publishedAt'][:10]
    #         start_date = video['insights']['startDate']
    #         time_diff = (datetime(*map(int, start_date.split('-'))) - datetime(*map(int, published_at.split('-')))).days
    #         days = read_as_int_array(video['insights']['days'], truncated=age) + time_diff
    #         days = days[days < age]
    #         daily_view = read_as_int_array(video['insights']['dailyView'], truncated=len(days))
    #         total_view = np.sum(daily_view)
    #         daily_watch = read_as_float_array(video['insights']['dailyWatch'], truncated=len(days))
    #         total_watch = np.sum(daily_watch)
    #         duration = isodate.parse_duration(video['contentDetails']['duration']).seconds
    #
    #         watch_at_age = total_watch * 60 / total_view / duration if total_view != 0 else None
    #         if 0 < watch_at_age <= 1:
    #             fout.write('{0}\t{1}\n'.format(duration, watch_at_age))

    for subdir, _, files in os.walk(data_loc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                for line in fin:
                    video = json.loads(line.rstrip())
                    if 'insights' not in video:
                        continue
                    if video['insights']['dailyWatch'] == 'N':
                        continue
                    published_at = video['snippet']['publishedAt'][:10]
                    start_date = video['insights']['startDate']
                    time_diff = (datetime(*map(int, start_date.split('-'))) - datetime(*map(int, published_at.split('-')))).days
                    days = read_as_int_array(video['insights']['days'], truncated=age) + time_diff
                    days = days[days < age]
                    daily_view = read_as_int_array(video['insights']['dailyView'], truncated=len(days))
                    total_view = np.sum(daily_view)
                    daily_watch = read_as_float_array(video['insights']['dailyWatch'], truncated=len(days))
                    total_watch = np.sum(daily_watch)
                    duration = isodate.parse_duration(video['contentDetails']['duration']).seconds

                    watch_at_age = total_watch*60/total_view/duration if (total_view != 0 and duration != 0) else None
                    if 0 < watch_at_age <= 1:
                        fout.write('{0}\t{1}\n'.format(duration, watch_at_age))

    fout.close()
