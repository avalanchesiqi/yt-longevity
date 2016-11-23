import os
from datetime import datetime
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdate


def flatten(arrs):
    """ Flatten list of arrays in to an array.
    """
    ret = []
    for arr in arrs:
        ret.extend(arr)
    return ret


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
            prev_rates = [monotonic_line[-1] for monotonic_line in monotonic_lines]
            m = len(prev_rates)
            for i, rate in enumerate(cur_rates):
                if rate <= min(prev_rates):
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
        ax.plot_date(ts_arr[5:], monotonic_line[5:], '-', ms=2, marker='x')

    xfmt = mdate.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_xlabel('Jun\' 19 2016 UTC')
    ax.set_title('Figure 4: Simulation result of fitting multiple lines.', y=-0.19)


if __name__ == '__main__':
    file_loc = '../../data/twitter-sampling'
    filename = '2016-10-14_inactive_rate.txt'
    filepath = os.path.join(file_loc, filename)

    ts_arr = [0]
    rate_arr = []
    raw_ts_arr = []
    raw_rate_arr = []
    with open(filepath, 'r') as filedata:
        for line in filedata:
            msg = json.loads(line.rstrip())
            ts = int(round(1.0 * int(msg['limit']['timestamp_ms']) / 1000))
            ts = datetime.utcfromtimestamp(ts)
            rate = int(msg['limit']['track'])
            if not ts_arr[-1] == ts:
                ts_arr.append(ts)
                rate_arr.append([rate])
            else:
                rate_arr[-1].append(rate)
            raw_ts_arr.append(ts)
            raw_rate_arr.append(rate)

    ts_arr.pop(0)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    plt.setp(ax1.get_xticklabels(), rotation=45)
    plt.setp(ax2.get_xticklabels(), rotation=45)

    plot_scatter(raw_ts_arr, raw_rate_arr, ax1)

    fit_and_plot_multiple_lines(rate_arr, ts_arr, ax2)

    # for i in xrange(int(mdate.date2num(ts_arr[0])), int(mdate.date2num(ts_arr[-1]))+1):
    #     if i not in ts_arr:
    #         print datetime.utcfromtimestamp(i)

    plt.show()
