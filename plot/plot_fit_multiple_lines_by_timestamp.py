import os
from datetime import datetime
import json
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.dates as mdate


def plot_scatter(arr1, arr2, ax):
    """ Scatter plot of track value in rate limit messages. """
    ax.scatter(arr1, arr2, marker='x', s=1, c='b')
    ax.set_xlim(xmin=min(arr1))
    ax.set_xlim(xmax=max(arr1))
    ax.set_ylim(ymin=min(arr2))
    ax.set_ylim(ymax=max(arr2))
    ax.set_xlabel('Jun\' 19 2016 UTC')
    ax.set_ylabel('Rate limit message value')
    ax.set_title('Figure 3: Scatter plot of rate limit messages.', y=-0.19)


def get_non_disconnect_window(arr, width):
    """ Get non-disconnect windows from rate limit messages. """
    box = deque(maxlen=width)
    windows = []
    for val in arr:
        if len(box) == 0:
            windows.append([val])
        elif len(box) < width and (min(box)-val) > 2500 and val < 300:
            # detect drop 1, not full, decrease sharply, hard threshold
            print 'detect drop #1: {0} {1} {2} {3}'.format(val, len(box), min(box), box)
            # reset box
            box = deque(maxlen=width)
            windows.append([val])
        elif len(box) == width and val < min(box) and val < 300:
            # detect drop 2, full, smaller than min in box, hard threshold
            print 'detect drop #2: {0} {1} {2} {3}'.format(val, len(box), min(box), box)
            # reset box
            box = deque(maxlen=width)
            windows.append([val])
        else:
            windows[-1].append(val)
        box.append(val)

    n1 = len(windows)
    print 'number of connections:', n1
    return windows


def fit_and_plot_multiple_lines(rate_arr, ts_arr, ax):
    """Separate an array into multiple monotonic arrays.
    """
    # plot simulation result of fitting multiple lines
    monotonic_lines = []
    for k, rates in enumerate(rate_arr):
        cur_rates = sorted(rates, reverse=True)
        if len(monotonic_lines) == 0:
            monotonic_lines = [[rate] for rate in cur_rates]
        else:
            if len(cur_rates) <= 4:
                prev_rates = [monotonic_line[-1] for monotonic_line in monotonic_lines]
                m = len(prev_rates)
                for i, rate in enumerate(cur_rates):
                    if rate <= min(prev_rates) and k < 100:
                        # fill with 0 backwards
                        monotonic_lines.append([0] * k)
                        monotonic_lines[-1].append(rate)
                    else:
                        for j in xrange(i, m):
                            if rate > prev_rates[j]:
                                monotonic_lines[j].append(rate)
                                prev_rates[j] = rate
                                break
                            elif len(monotonic_lines[j]) < k:
                                monotonic_lines[j].append(prev_rates[j])

            for monotonic_line in monotonic_lines:
                if not len(monotonic_line) == k+1:
                    monotonic_line.append(monotonic_line[-1])

    for monotonic_line in monotonic_lines:
        ax.plot_date(ts_arr, monotonic_line, '-', ms=2, marker='x')

    xfmt = mdate.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_xlabel('Jun\' 19 2016 UTC')
    ax.set_title('Figure 4: Simulation result of fitting multiple lines.', y=-0.19)


if __name__ == '__main__':
    file_loc = '../../data/twitter-sampling'
    filename = 'dml_rate.txt'
    filepath = os.path.join(file_loc, filename)

    raw_ts_arr = []
    raw_rate_arr = []
    with open(filepath, 'r') as filedata:
        for line in filedata:
            msg = json.loads(line.rstrip())
            ts = int(round(1.0 * int(msg['limit']['timestamp_ms']) / 1000))
            ts = datetime.utcfromtimestamp(ts)
            rate = int(msg['limit']['track'])
            raw_ts_arr.append(ts)
            raw_rate_arr.append(rate)

    width = 20
    non_disconnect_windows = get_non_disconnect_window(raw_rate_arr, width)

    length_arr = [len(window) for window in non_disconnect_windows]
    non_disconnect_timestamps = []
    prev_index = 0
    for i in xrange(len(length_arr)):
        non_disconnect_timestamps.append(raw_ts_arr[prev_index: prev_index+length_arr[i]])
        prev_index += length_arr[i]

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

    plot_scatter(raw_ts_arr, raw_rate_arr, ax1)

    for non_disconnect_timestamp, non_disconnect_window in zip(non_disconnect_timestamps, non_disconnect_windows):
        ts_arr = [0]
        rate_arr = []
        for i in xrange(len(non_disconnect_window)):
            ts = non_disconnect_timestamp[i]
            rate = non_disconnect_window[i]
            if not ts_arr[-1] == ts:
                ts_arr.append(ts)
                rate_arr.append([rate])
            else:
                rate_arr[-1].append(rate)
        ts_arr.pop(0)
        fit_and_plot_multiple_lines(rate_arr, ts_arr, ax2)

    plt.setp(ax1.get_xticklabels(), rotation=45)
    plt.setp(ax2.get_xticklabels(), rotation=45)

    plt.show()
