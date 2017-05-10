#!/usr/bin/env python

import sys
import os
import operator
import argparse
import numpy as np
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def millions(x, pos):
    return '{0:.2f}M'.format(x/1000000)

m_formatter = FuncFormatter(millions)


def read_as_int_array(content):
    return np.array(map(int, content.split(',')), dtype=np.uint32)


def plot_trend(ax, video, upload_dates):
    ax_y = video.get_dailyview()
    start_date = video.get_start_date()
    ax_x = [start_date + timedelta(days=i) for i in xrange(len(ax_y))]
    ax.plot_date(ax_x, ax_y, '-', c='b')
    ax.plot((start_date, start_date), (0, ax.get_ylim()[1]), 'g-')
    for date in upload_dates:
        if (date-start_date).days > 0:
            ax.plot((date, date), (0, ax.get_ylim()[1]), 'r-')
    ax.yaxis.set_ticks([0, ax.get_ylim()[1]])
    ax.get_yaxis().set_major_formatter(m_formatter)
    ax.text(0.01, 0.95, '{0}: {1:.2e}'.format(video.get_title(), video.get_totalview()), transform=ax.transAxes, fontsize=12, verticalalignment='top')


class Video:
    def __init__(self, content):
        self.content = content

    def __hash__(self):
        return hash(self.content['id'])

    def __eq__(self, other):
        return self.content['id'] == other.content['id']

    def get_days(self):
        return read_as_int_array(self.content['insights']['days'])

    def get_dailyview(self):
        return read_as_int_array(self.content['insights']['dailyView'])

    def get_totalview(self):
        return int(self.content['statistics']['viewCount'])

    def get_start_date(self):
        return datetime(*map(int, self.content['insights']['startDate'].split('-')))

    def get_title(self):
        return self.content['snippet']['title']


def plot_main(input_path):
    video_view_dict = {}
    with open(input_path, 'r') as f:
        for line in f:
            video_json = json.loads(line.rstrip())
            video_obj = Video(video_json)
            total_view = video_obj.get_totalview()
            video_view_dict[video_obj] = total_view
    sorted_by_view = sorted(video_view_dict.items(), key=operator.itemgetter(1), reverse=True)[:10]
    videos = [t[0] for t in sorted_by_view]

    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(10, 1, sharex=True)
    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]

    upload_dates = [video.get_start_date() for video in videos]
    for idx, video in enumerate(videos):
        plot_trend(axs[idx], video, upload_dates)


if __name__ == '__main__':
    # I/O interface, read from an input file
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file path of video metadata/insight data', required=True)
    args = parser.parse_args()
    input_path = args.input

    plot_main(input_path)

    plt.tight_layout()
    plt.show()
