import os
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import logging

file_loc = '../../data/rate_limit'


def list_format(lst):
    res = ''
    for l in lst:
        res += str(l)[1:-1]
        res += '-'
    return res


def plot_drop_and_fit(path, width):
    box = deque(maxlen=width)
    lines = []
    with open(path, 'r') as data:
        for line in data:
            track = int(line.rstrip())
            if len(box) == 0:
                lines.append([track])
            elif len(box) < width:
                lines[-1].append(track)
            else:
                drop_bound = min(box)
                if track < drop_bound and track < 100:
                    # detect drop
                    # print 'detect drop', track, drop_bound, box
                    box = deque(maxlen=width)
                    lines.append([track])
                else:
                    lines[-1].append(track)
            box.append(track)

    n = len(lines)
    logging.debug('{0}:{1}'.format(path[27: 37], list_format(lines)))
    print path[27: 37], 'number of connections', n
    return path[27:37], n-1
    # prev = 0
    # for i in lines:
    #     print len(i)
    #     ax1.scatter(prev+np.arange(len(i)), i, s=2)
    #     prev += len(i)


def date2str(date_format):
    return date_format.strftime("%Y-%m-%d")


if __name__ == '__main__':
    box_width = 50
    disconnect_map = {}
    fig, (ax1, ax2) = plt.subplots(2, 1)

    logging.basicConfig(filename='../../data/test_sampling/rate_limit_msgs.log', level=logging.DEBUG)

    for subdir, _, files in os.walk(file_loc):
        for f in sorted(files):
            filepath = os.path.join(subdir, f)
            date, disconnect = plot_drop_and_fit(filepath, box_width)
            disconnect_map[date] = disconnect

    # filepath = os.path.join(file_loc, 'rate_2015-12-16.bz2.txt')
    # plot_drop_and_fit(filepath, box_width)

    date_axis = [datetime(2014, 5, 28)+timedelta(days=i) for i in xrange(882)]
    disconnect_axis = [disconnect_map[date2str(d)] if date2str(d) in disconnect_map else np.nan for d in date_axis]
    ax1.plot_date(date_axis, disconnect_axis, '-', ms=2, color='b')
    ax1.bar(date_axis, np.isnan(disconnect_axis)*ax1.get_ylim()[1], color=(.9, .9, .9))
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Number of disconnect/reconnect')

    disconnect_axis = [i for i in disconnect_axis if i is not np.nan]
    weights = 1.0 * np.ones_like(disconnect_axis) / len(disconnect_axis)
    ax2.hist(disconnect_axis, weights=weights, bins=30)
    ax2.set_xlabel('Number of disconnect/reconnect')
    ax2.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

