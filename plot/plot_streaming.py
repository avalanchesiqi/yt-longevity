#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def plot_main(filepath, year, ax, c, name):
    if year == '2014' or year == 'rate_2014':
        starttime = datetime(2014, 6, 27)
    # if year == '2015':
    else:
        starttime = datetime(2015, 6, 27)
    time_axis = [starttime + timedelta(minutes=i) for i in xrange(1440*3)]
    data_axis = []
    cnt_lst = []
    for subdir, _, files in os.walk(filepath):
        for f in sorted(files):
            if f.startswith(year):
                filename = os.path.join(subdir, f)
                with open(filename, 'r') as filedata:
                    cnt = 0
                    for line in filedata:
                        data_axis.append(int(line.rstrip()))
                        cnt += int(line.rstrip())
                cnt_lst.append(cnt)
    ax.plot_date(time_axis, data_axis, '-', color=c, label='{0}: {1},{2}'.format(name, ','.join(map(str, cnt_lst)), sum(data_axis)))
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M:%S'))
    ax.set_ylim(ymax=14000)
    ax.set_ylim(ymin=0)
    ax.legend(loc='best')
    return data_axis


if __name__ == '__main__':
    # filepath1 = '../../data/twitter_stream_archived/'
    filepath1 = '../../data/yt_crawler/'

    dict_lst = []
    for subdir, _, files in os.walk(filepath1):
        for f in files:
            if f.startswith('tweet'):
                filepath = os.path.join(subdir, f)
                print filepath
                tmp = json.load(open(filepath))
                print tmp
                dict_lst.append(tmp)

    stream_dict = merge_dicts(*dict_lst)
    for k, v in stream_dict.items():
        print k
        print v


    # fig, ax = plt.subplots(1, 1)
    # as2014 = plot_main(filepath1, '2014', ax1, 'b', 'archived stream')
    #
    # time_axis1 = [datetime(2014, 6, 1) + timedelta(minutes=i) for i in xrange(1440 * 30)]
    #
    # for i in xrange(1440 * 3):
    #     if as2014[i] < yt2014[i]:
    #         ax1.axvline(time_axis1[i], color=(.5, .5, .5), zorder=0, linewidth=0.2)
    # for i in xrange(1440 * 3):
    #     if as2015[i] < yt2015[i]:
    #         ax2.axvline(time_axis2[i], color=(.5, .5, .5), zorder=0, linewidth=0.2)
    #
    # plot_main(filepath2, 'rate_2014', ax1, 'g', 'rate limit msg')
    # plot_main(filepath2, 'rate_2015', ax2, 'g', 'rate limit msg')
    #
    # plt.show()
