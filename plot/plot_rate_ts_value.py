import os
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

file_loc = '../../data/rate_limit'


def fit_multiple_lines(lst, head=0, prev=0):
    lines = [[lst[0]]]
    orders = [[1+prev]]
    n = len(lst)
    cnt = 2+prev
    for i in xrange(1, n):
        comer = lst[i]
        m = len(lines)
        last_elms = np.array([lines[j][-1] for j in xrange(m)])
        if comer <= min(last_elms):
            if head == 0 or comer < 100:
                lines.append([comer])
                orders.append([cnt])
        else:
            last_elms[comer <= last_elms] = 0
            k = np.argmin(comer-last_elms)
            # detect unusual increment
            lines[k].append(comer)
            orders[k].append(cnt)
        cnt += 1
    t = len(lines)
    print 'number of inner segments:', t, len(orders)
    if head == 0:
        miss_num = sum([(lines[i][-1] - lines[i][0]) for i in xrange(t)])
    else:
        miss_num = sum([(lines[i][-1]) for i in xrange(t)])
    return orders, lines, miss_num


def get_miss_num_sum(arrs, ax):
    prev = 0
    total_miss = 0
    for (cnt, arr) in enumerate(arrs):
        orders, segs, segs_miss_num = fit_multiple_lines(arr, head=cnt, prev=prev)
        prev += sum([len(order) for order in orders])
        total_miss += segs_miss_num
        for order, seg in zip(orders, segs):
            ax.plot(order, seg, '-', ms=2, marker='x')
        ax.set_title('Simulation result of assigning rate limit message to nearby line')
        ax.set_xlabel('Incoming order')
    print 'sum total miss:', total_miss


def get_disconnect_window(arr, width):
    box = deque(maxlen=width)
    windows = []
    for val in arr:
        if len(box) == 0:
            windows.append([val])
        elif len(box) < width and (min(box)-val) > 2500 and val < 100:
            # detect drop 1, not full, decrease sharply, hard threshold
            print 'detect drop #1: {0} {1} {2} {3}'.format(val, len(box), min(box), box)
            # reset box
            box = deque(maxlen=width)
            windows.append([val])
        elif len(box) == width and min(box) > val and val < 100:
            # detect drop 2, full, smaller than min in box, hard threshold
            print 'detect drop #2: {0} {1} {2} {3}'.format(val, len(box), min(box), box)
            # reset box
            box = deque(maxlen=width)
            windows.append([val])
        else:
            windows[-1].append(val)
        box.append(val)

    n = len(windows)
    print 'number of connections:', n
    return windows


def plot_track_scatter(arr, ax):
    """ Scatter plot of value in rate limit messages"""
    xaxis = np.arange(1, len(arr)+1)
    ax.scatter(xaxis, arr, marker='x', s=1, c='b')
    ax.set_title('Scatter plot of incoming order wrt rate limit message value')
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymin=0)
    ax.set_xlabel('Incoming order')
    ax.set_ylabel('Rate limit value')


if __name__ == '__main__':
    # 2015-04-09
    filepath = os.path.join(file_loc, 'rate_2016-07-21.bz2.txt')

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    tracks = []
    with open(filepath, 'r') as filedata:
        for line in filedata:
            track = int(line.rstrip())
            tracks.append(track)
    plot_track_scatter(tracks, ax1)

    width = 50
    non_disconnect_windows = get_disconnect_window(tracks, width)
    get_miss_num_sum(non_disconnect_windows, ax2)

    plt.show()
