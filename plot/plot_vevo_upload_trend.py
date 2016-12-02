import os
from collections import defaultdict
import json
from dateutil import parser
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    file_loc = '../../data/vevo'
    upload_dict = defaultdict(int)
    upload_artist_dict = defaultdict(lambda: set())

    for subdir, _, files in os.walk(file_loc):
        for f in files:
            filepath = os.path.join(subdir, f)
            cnt = 0
            with open(filepath, 'r') as filedata:
                for line in filedata:
                    if line.rstrip():
                        video = json.loads(line.rstrip())
                        published_at = video['snippet']['publishedAt']
                        dt = parser.parse(published_at)
                        mth_dt = datetime(dt.year, dt.month, 1)
                        upload_dict[mth_dt] += 1
                        channel_id = video['snippet']['channelId']
                        upload_artist_dict[mth_dt].add(channel_id)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    x_axis = sorted(upload_dict.keys())
    upload_axis = [upload_dict[k] for k in x_axis]
    upload_artist_axis = [len(upload_artist_dict[k]) for k in x_axis]
    upload_avg_axis = [1.0*upload_dict[k]/len(upload_artist_dict[k]) for k in x_axis]

    ax1.plot_date(x_axis, upload_axis, 'o-', c='b')
    ax2.plot_date(x_axis, upload_artist_axis, 'o-', c='b')
    ax3.plot_date(x_axis, upload_avg_axis, 'o-', c='b')

    # ax1.set_xscale('log')
    # ax1.set_yscale('log')
    # ax1.set_xlim(xmin=1)
    # ax1.set_ylim(ymin=1)
    ax1.set_ylabel('Number of VEVO videos published')
    ax1.set_title('Figure 1: Trend of VEVO video publishing.')

    ax2.set_ylabel('Number of artists publishing video')
    ax2.set_title('Figure 2: Trend of VEVO artist publish.')

    ax3.set_xlabel('Calender month')
    ax3.set_ylabel('Number of video per VEVO artist')
    ax3.set_title('Figure 3: Trend of video per VEVO artist.')

    plt.show()
