import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from sklearn import linear_model

file_loc = '../../data/test_sampling'


def divide_fit(X, y, ax):
    n = len(X)
    X1 = X[: n/2]
    X2 = X[n/2:]
    y1 = y[: n/2]
    y2 = y[n/2:]
    lr1 = linear_model.LinearRegression()
    lr1.fit(X1, y1)
    if lr1.coef_ < 0:
        divide_fit(X1, y1, ax)
    else:
        X_ts_track1 = mdate.epoch2num(np.array(X1) / 1000)
        ax.plot_date(X_ts_track1, lr1.predict(np.array(X1).reshape(-1, 1))[:], marker='o', ms=1, c='red')
    lr2 = linear_model.LinearRegression()
    lr2.fit(X2, y2)
    if lr2.coef_ < 0:
        divide_fit(X2, y2, ax)
    else:
        X_ts_track2 = mdate.epoch2num(np.array(X2)/1000)
        ax.plot_date(X_ts_track2, lr2.predict(np.array(X2).reshape(-1, 1))[:], marker='o', ms=1, c='red')

def plot_ts_value(values, tss, ax1, ax2):
    # Choose your xtick format string
    date_fmt = '%d-%m-%y %H:%M:%S'
    # Use a DateFormatter to set the data to the correct format.
    date_formatter = mdate.DateFormatter(date_fmt)
    ax1.xaxis.set_major_formatter(date_formatter)
    ax2.xaxis.set_major_formatter(date_formatter)
    # Sets the tick labels diagonal so they fit easier.
    fig.autofmt_xdate()

    ts_track = np.array(tss) / 1000
    ts_track = mdate.epoch2num(ts_track)

    ax1.plot_date(ts_track, values, marker='x', ms=1, c='green')

    n = len(tss)
    gap = 1000
    k = n/gap
    for i in xrange(k+1):
        X = np.array(tss[i*gap: min((i+1)*gap, n)]).reshape(-1, 1)
        y = np.array(values[i*gap: (i+1)*gap]).reshape(-1, 1)
        lr = linear_model.LinearRegression()
        lr.fit(X, y)

        if lr.coef_ < 0:
            divide_fit(X, y, ax2)
        else:
            X_ts_track = mdate.epoch2num(np.array(X)/1000)
            ax2.plot_date(X_ts_track, lr.predict(np.array(X).reshape(-1, 1))[:], marker='o', ms=1, c='red')

    ax1.set_ylim(ymin=0)
    ax2.set_ylim(ymin=0)

    ax1.set_xlabel('Epoch time')
    ax2.set_xlabel('Epoch time')
    ax1.set_ylabel('Rate limit track')
    ax2.set_ylabel('Rate limit track')


def read_data(filedata):
    values = map(int, filedata.readline().rstrip().split(':')[2].split(','))
    tss = map(int, filedata.readline().rstrip().split(':')[2].split(','))
    # ts_track = np.array(map(int, data.readline().rstrip().split(':')[2].split(','))) / 1000
    # ts_track = mdate.epoch2num(ts_track)
    return values, tss

if __name__ == '__main__':
    filepath = os.path.join(file_loc, 'maximum_invert_ts.log')

    fig = plt.figure()
    ax1 = fig.add_subplot(241)
    ax2 = fig.add_subplot(242)
    ax3 = fig.add_subplot(243)
    ax4 = fig.add_subplot(244)
    ax5 = fig.add_subplot(245)
    ax6 = fig.add_subplot(246)
    ax7 = fig.add_subplot(247)
    ax8 = fig.add_subplot(248)
    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
    with open(filepath, 'r') as filedata:
        cnt = 0
        s = 0
        while cnt < 4:
            values, tss = read_data(filedata)
            if s > 20:
                plot_ts_value(values, tss, axs[cnt], axs[cnt+4])
                cnt += 1
            s += 1

    plt.show()
