#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import isodate
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

category_dict = {"42": "Shorts", "29": "Nonprofits & Activism", "24": "Entertainment", "25": "News & Politics", "26": "Howto & Style", "27": "Education", "20": "Gaming", "21": "Videoblogging", "22": "People & Blogs", "23": "Comedy", "44": "Trailers", "28": "Science & Technology", "43": "Shows", "40": "Sci-Fi/Fantasy", "41": "Thriller", "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music", "39": "Horror", "38": "Foreign", "15": "Pets & Animals", "17": "Sports", "19": "Travel & Events", "18": "Short Movies", "31": "Anime/Animation", "30": "Movies", "37": "Family", "36": "Drama", "35": "Documentary", "34": "Comedy", "33": "Classics", "32": "Action/Adventure"}


def x_fmt(x, p):
    return int(x/6)


def plot_category(category_id):
    data_loc = '../../data/byCategory/{0}.json'.format(category_id)

    duration_dict = defaultdict(int)
    with open(data_loc, 'r') as filedata:
        for line in filedata:
            video = json.loads(line.rstrip())
            if video['insights']['avgWatch'] == 'N':
                continue
            published_at = video['snippet']['publishedAt'][:10]
            if published_at[:4] == '2016':
                duration = isodate.parse_duration(video['contentDetails']['duration']).seconds
                avg_watch = float(video['insights']['avgWatch']) * 60
                if duration > 0:
                    if avg_watch <= duration < 3600:
                        duration_dict[duration/10] += 1

    x_axis = sorted(duration_dict.keys())
    y_axis = [duration_dict[x] for x in x_axis]
    ax1.plot(x_axis, y_axis, label='{0}'.format(category_dict[category_id]))


if __name__ == '__main__':
    category_ids = ['10', '20', '22', '24', '25']
    fig, ax1 = plt.subplots(1, 1)

    for cid in category_ids:
        plot_category(cid)

    ax1.set_ylim(ymin=0)
    ax1.set_xlim(xmin=0)
    ax1.set_xlim(xmax=360)
    ax1.get_xaxis().set_major_formatter(FuncFormatter(x_fmt))
    ax1.set_xlabel('video duration (minute)')
    ax1.set_ylabel('video number')

    plt.legend(loc='upper right')
    plt.show()
