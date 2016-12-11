import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import dateutil.parser


def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.0fM' % (x * 1e-6)

m_formatter = FuncFormatter(millions)


def thousands(x, pos):
    'The two args are the value and tick position'
    return '%1.0fk' % (x * 1e-3)

k_formatter = FuncFormatter(thousands)


if __name__ == '__main__':
    file_loc = '../data/news'
    fig, (ax1, ax2) = plt.subplots(1, 2)
    labels = ['ABC News', 'BBC News', 'Alex Jones', 'Young Turks', 'RT']
    view_matrix = []
    comment_matrix = []

    for subdir, _, files in os.walk(file_loc):
        for f in files:
            filepath = os.path.join(subdir, f)
            view_list = []
            comment_list = []
            cnt = 0
            dump = 0
            with open(filepath, 'r') as filedata:
                for line in filedata:
                    if line.rstrip():
                        video = json.loads(line.rstrip())
                        channel_title = video['snippet']['channelTitle']
                        dt = dateutil.parser.parse(video['snippet']['publishedAt']).replace(tzinfo=None)
                        if dt > datetime(2016, 7, 31, 23, 59, 59):
                            cnt += 1
                        try:
                            view_count = int(video['statistics']['viewCount'])
                            view_list.append(view_count)
                        except:
                            dump += 1
                        try:
                            comment_count = int(video['statistics']['commentCount'])
                            comment_list.append(comment_count)
                        except:
                            dump += 1

            view_matrix.append(view_list)
            comment_matrix.append(comment_list)

    ax1.boxplot(view_matrix, labels=labels, showmeans=True, showfliers=False)
    ax2.boxplot(comment_matrix, labels=labels, showmeans=True, showfliers=False)

    ax1.yaxis.set_major_formatter(k_formatter)
    ax1.set_ylabel('Number of lifetime views')
    ax1.set_title('Figure 4: Lifetime views boxplot')

    ax2.yaxis.set_major_formatter(k_formatter)
    ax2.set_ylabel('Number of lifetime comments')
    ax2.set_title('Figure 5: Lifetime comments boxplot')

    plt.show()
