#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy import stats
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # vid, duration, view, watchtime, c
    # 3.2% videos have watch percentage over 1, 357 out of 10915
    data_loc = '../../rdata/data.csv'

    fig, ax1 = plt.subplots(1, 1)

    x_axis = []
    y_axis = []

    with open(data_loc, 'r') as f1:
        for line in f1:
            vid, duration, view, watchtime, c = line.rstrip().split()
            duration = int(duration)
            view = int(view)
            watchtime = float(watchtime)
            c = float(c)
            if view > 0 and duration > 0:
                wp = watchtime*60/view/duration
                if wp > 1:
                    wp = 1
                x_axis.append(wp)
                y_axis.append(c)

    print stats.pearsonr(x_axis, y_axis)
    print stats.kendalltau(x_axis, y_axis)

    ax1.scatter(x_axis, y_axis, lw=0)
    ax1.set_xlabel('watch percentage')
    ax1.set_ylabel('c value')
    ax1.set_xlim(xmin=0)
    ax1.set_ylim(ymin=0)
    ax1.set_xlim(xmax=1)
    ax1.set_yscale('log')

    plt.show()
