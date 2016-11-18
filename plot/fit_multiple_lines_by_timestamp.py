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


def fit_and_plot_multiple_lines(rate_arr, ts_arr, ax):
    """Separate an array into multiple monotonic arrays.
    """
    # plot simulation result of fitting multiple lines
    monotonic_lines = []
    for k, rates in enumerate(rate_arr):
        sorted_rates = sorted(rates, reverse=True)
        if len(monotonic_lines) == 0:
            previous_rates = sorted_rates
            monotonic_lines = [[rate] for rate in sorted_rates]
        else:
            previous_rates = [monotonic_line[-1] for monotonic_line in monotonic_lines]
            m = len(previous_rates)
            n = len(sorted_rates)
            if n > m:
                for i in xrange(n-m):
                    monotonic_lines.append([sorted_rates.pop(n-1-i)])
            for cnt, rate in enumerate(sorted_rates):
                if rate <= min(previous_rates):
                    monotonic_lines.append([rate])
                else:
                    for i in xrange(m):
                        if rate > previous_rates[i]:
                            monotonic_lines[i].append(rate)
                            previous_rates[i] = rate
                            break

        # fulfill with the last element
        n = max([len(monotonic_line) for monotonic_line in monotonic_lines])
        for monotonic_line in monotonic_lines:
            if len(monotonic_line) < n:
                dup_rate = monotonic_line[-1]
                for _ in xrange(n-len(monotonic_line)):
                    monotonic_line.append(dup_rate)

    ts_arr = [datetime.utcfromtimestamp(ts) for ts in ts_arr]
    for monotonic_line in monotonic_lines:
        ax.plot_date(ts_arr, monotonic_line, '-', ms=2, marker='x')

    xfmt = mdate.DateFormatter('%Y-%m-%d %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_title('Simulation result of fitting multiple lines')
    ax.set_xlabel('UTC datetime')


if __name__ == '__main__':
    file_loc = '../../data'
    filename = '2016-06-23-2.txt'
    filepath = os.path.join(file_loc, filename)

    rate_arr = []
    ts_arr = [0]
    with open(filepath, 'r') as filedata:
        for line in filedata:
            msg = json.loads(line.rstrip())
            rate = int(msg['limit']['track'])
            ts = int(round(1.0*int(msg['limit']['timestamp_ms'])/1000))
            if not ts_arr[-1] == ts:
                rate_arr.append([rate])
                ts_arr.append(ts)
            else:
                rate_arr[-1].append(rate)

    ts_arr.pop(0)
    fig, ax1 = plt.subplots(1, 1)

    fit_and_plot_multiple_lines(rate_arr, ts_arr, ax1)

    # for i in xrange(ts_arr[0], ts_arr[-1]+1):
    #     if i not in ts_arr:
    #         print datetime.utcfromtimestamp(i)

    plt.show()
