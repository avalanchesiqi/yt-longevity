import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdate

filepath = '../../data/test_sampling/maximum_invert_ts.log'


def plot_max_inv(data, ts, ax):
    n = len(data)
    res = [[data[0]]]
    order = [[ts[0]]]
    for i in xrange(1, n):
        m = len(res)
        arr = [res[m-1-v][-1] if (m-1-v >= 0 and data[i] > res[m-1-v][-1]) else -np.inf for v in xrange([4, m%4][m%4>0])]
        if sum(np.isfinite(arr)):
            j = np.argmax(arr)
            res[m-1-j].append(data[i])
            order[m-1-j].append(ts[i])
        else:
            res.append([data[i]])
            order.append([ts[i]])
    k = len(res)
    # print 'number of lines: {0}'.format(k)
    # Choose your xtick format string
    date_fmt = '%d-%m-%y %H:%M:%S'

    # Use a DateFormatter to set the data to the correct format.
    date_formatter = mdate.DateFormatter(date_fmt)
    ax.xaxis.set_major_formatter(date_formatter)

    # Sets the tick labels diagonal so they fit easier.
    fig.autofmt_xdate()
    for t in xrange(k):
        ax.plot_date(order[t], res[t], ls='--', marker='x', ms=2)
    ax.set_xlabel('Epoch time')
    ax.set_ylabel('Rate limit track')
    ax.text(0.55, 0.95, 'number of lines: {0}'.format(k), transform=ax.transAxes, fontsize=14, verticalalignment='top')
    plt.setp(ax.get_xticklabels(), visible=True)


def read_data():
    miss_track = map(int, data.readline().rstrip().split(':')[2].split(','))
    ts_track = np.array(map(int, data.readline().rstrip().split(':')[2].split(','))) / 1000
    ts_track = mdate.epoch2num(ts_track)
    return miss_track, ts_track

if __name__ == '__main__':
    fig = plt.figure()
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
    axs = [ax1, ax2, ax3, ax4, ax5, ax6]
    with open(filepath, 'r') as data:
        cnt = 0
        s = 0
        while cnt < 6:
            miss_track, ts_track = read_data()
            if s > 9:
                plot_max_inv(miss_track, ts_track, axs[cnt])
                cnt += 1
            s += 1

    plt.show()
