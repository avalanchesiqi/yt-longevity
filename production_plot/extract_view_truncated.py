#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import sys
import os
import json
import numpy as np
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt


def read_as_int_array(content, truncated=None):
    if truncated is None:
        return np.array(map(int, content.split(',')), dtype=np.uint32)
    else:
        return np.array(map(int, content.split(',')), dtype=np.uint32)[:truncated]


def batch_write(matrix):
    matrix = matrix.T
    for i in xrange(matrix.shape[0]):
        with open('../../data/production_data/{0}.txt'.format(i), 'a') as fout:
            for j in matrix[i]:
                fout.write('{}\n'.format(j))
    return

if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    age = 182
    view_percent_matrix = None
    cnt = 0

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    data_loc = '../../data/production_data/random_dataset/'

    # == == == == == == == == Part 3: Read dataset and update matrix == == == == == == == == #
    for subdir, _, files in os.walk(data_loc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as filedata:
                for line in filedata:
                    if cnt > 10000:
                        batch_write(view_percent_matrix)
                        view_percent_matrix = None
                        cnt = 0
                    video = json.loads(line.rstrip())
                    published_at = video['snippet']['publishedAt'][:10]
                    start_date = video['insights']['startDate']
                    time_diff = (datetime(*map(int, start_date.split('-'))) - datetime(*map(int, published_at.split('-')))).days
                    days = read_as_int_array(video['insights']['days'], truncated=age) + time_diff
                    days = days[days < age]
                    daily_view = read_as_int_array(video['insights']['dailyView'], truncated=len(days))
                    total_view = np.sum(daily_view)

                    # when view statistic is missing, fill 0s
                    filled_view_percent = np.zeros(age)
                    filled_view_percent[days] = daily_view/total_view if total_view != 0 else 0
                    if view_percent_matrix is None:
                        view_percent_matrix = filled_view_percent
                    else:
                        view_percent_matrix = np.vstack((view_percent_matrix, filled_view_percent))
                    cnt += 1
