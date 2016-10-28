import os
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import logging

file_loc = '../../data/test_sampling'


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
    fig, ax1 = plt.subplots(1, 1)

    filepath = os.path.join(file_loc, 'inner_segments.log')
    dist = []

    with open(filepath, 'r') as filedata:
        nextline = filedata.readline()
        while nextline:
            try:
                num = int(nextline.rstrip().rsplit(':', 1)[1])
                dist.append(num)
                if not num == 4:
                    filedata.readline()
                nextline = filedata.readline()
            except Exception as e:
                print str(e)

    weights = 1.0 * np.ones_like(dist) / len(dist)
    ax1.hist(dist, weights=weights, bins=max(dist))
    ax1.set_xlabel('Number of segments')
    ax1.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

