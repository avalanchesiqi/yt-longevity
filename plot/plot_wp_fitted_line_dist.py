#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import numpy as np
import isodate
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# if __name__ == '__main__':
#     data_loc = '../linux_figs/detailed_videos/'
#
#     fig = plt.figure()
#
#     coef_list = []
#     intercept_list = []
#
#     for subdir, _, files in os.walk(data_loc):
#         for f in files:
#             if f.endswith('txt'):
#                 filepath = os.path.join(subdir, f)
#                 with open(filepath, 'r') as fdata:
#                     for line in fdata:
#                         _, coef, intercept, _ = line.rstrip().split()
#                         coef = float(coef)
#                         intercept = float(intercept)
#                         coef_list.append(coef)
#                         intercept_list.append(intercept)
#
#     ax1 = fig.add_subplot(121)
#     ax2 = fig.add_subplot(122)
#
#     ax1.hist(coef_list, 10)
#     ax2.hist(intercept_list, 10)
#
#     print np.mean(coef_list)
#     print np.mean(intercept_list)
#
#     plt.show()

from collections import Counter

if __name__ == '__main__':
    data_loc = '../../data/byCategory/'

    fig = plt.figure()

    coef_list = []
    intercept_list = []

    lst = []

    for subdir, _, files in os.walk(data_loc):
        for f in files:
            filepath = os.path.join(subdir, f)
            with open(filepath, 'r') as fdata:
                for line in fdata:
                    video = json.loads(line.rstrip())
                    duration = isodate.parse_duration(video['contentDetails']['duration']).seconds
                    if 'avgWatch' in video['insights'] and duration > 0:
                        avgWatch = video['insights']['avgWatch']
                        if not avgWatch == 'N':
                            wp = float(avgWatch)*60/duration
                            if 0 < wp < 1:
                                lst.append(int(wp*100/10))
                            elif wp >= 1:
                                lst.append(9)

    ax1 = fig.add_subplot(111)
    # ax2 = fig.add_subplot(122)

    ax1.hist(lst, bins=10)
    ax1.set_xlim(xmax=10)
    ax1.set_xlim(xmin=0)

    print Counter(lst)
    # print np.mean(lst)
    # ax2.hist(intercept_list, 10)

    # print np.mean(coef_list)
    # print np.mean(intercept_list)

    plt.show()