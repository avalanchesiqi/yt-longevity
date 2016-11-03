#!/usr/bin/env python

import sys
import os
import operator
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def millions(x, pos):
    return '{0:.2f}M'.format(x/1000000)

m_formatter = FuncFormatter(millions)


def plot_trend(path, idx, dates):
    with open(path, 'r') as data:
        vid = path.rsplit('\\', 1)[1]
        start_date, dailyview, totalview, dailyshare, totalshare = data.readline().rstrip().split()[:5]
        y, m, d = map(int, start_date.split('-'))
        start_date = datetime(y, m, d)
        ax_y = map(int, dailyview.split(','))
        ax_x = [start_date + timedelta(days=i) for i in xrange(len(ax_y))]
        current_ax = axs[idx]
        current_ax.plot_date(ax_x, ax_y, '-', c='b')
        current_ax.plot((start_date, start_date), (0, current_ax.get_ylim()[1]), 'g-')
        for date in dates:
            if (date-start_date).days > 0:
                current_ax.plot((date, date), (0, current_ax.get_ylim()[1]), 'r-')
        current_ax.yaxis.set_ticks([0, current_ax.get_ylim()[1]])
        current_ax.get_yaxis().set_major_formatter(m_formatter)
        current_ax.text(0.01, 0.95, '{0}: {1:.2E}'.format(vid_title_dict[vid], int(totalview)), transform=axs[idx].transAxes, fontsize=12, verticalalignment='top')


def get_top10_paths(dir_path):
    path_view_dict = {}
    for subdir, _, files in os.walk(dir_path):
        for f in files:
            file_path = os.path.join(subdir, f)
            with open(file_path, 'r') as data:
                _, _, totalview = data.readline().rstrip().split()[:3]
                path_view_dict[file_path] = int(totalview)
    sorted_by_view = sorted(path_view_dict.items(), key=operator.itemgetter(1), reverse=True)[:10]
    paths = [t[0] for t in sorted_by_view]
    return paths


def get_dates(path):
    with open(path, 'r') as data:
        start_date = data.readline().rstrip().split()[0]
        y, m, d = map(int, start_date.split('-'))
        start_date = datetime(y, m, d)
    return start_date


def plot_main(dir_path):
    top10_paths = get_top10_paths(dir_path)
    dates = []
    for path in top10_paths:
        dates.append(get_dates(path))
    for idx, path in enumerate(top10_paths):
        plot_trend(path, idx, dates)


def get_vid_title(path):
    vid_title_dict = {}
    with open(path, 'r') as data:
        for line in data:
            vid, _, _, title = line.rstrip().split(',', 3)
            vid_title_dict[vid] = title
    return vid_title_dict


if __name__ == '__main__':
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(10, 1, sharex=True)
    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]
    channel_title = 'KatyPerryVEVO'
    dir_path = '../output/{0}'.format(channel_title)
    vid_title_dict = get_vid_title('../input/{0}.txt'.format(channel_title))

    plot_main(dir_path)
    plt.tight_layout()
    plt.show()
