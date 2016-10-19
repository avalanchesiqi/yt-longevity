import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from sklearn import linear_model

file_loc = '../../data/test_sampling'


def plot_ts_value(values, tss, ax):
    # Choose your xtick format string
    date_fmt = '%d-%m-%y %H:%M:%S'
    # Use a DateFormatter to set the data to the correct format.
    date_formatter = mdate.DateFormatter(date_fmt)
    ax.xaxis.set_major_formatter(date_formatter)
    # Sets the tick labels diagonal so they fit easier.
    fig.autofmt_xdate()

    ts_track = np.array(tss) / 1000
    ts_track = mdate.epoch2num(ts_track)

    ax.plot_date(ts_track, values, marker='x', ms=1, c='green')

    n = len(tss)
    gap = 1000
    k = n/gap
    for i in xrange(k+1):
        lr = linear_model.LinearRegression()
        X = np.array(tss[i*gap: min((i+1)*gap, n)]).reshape(-1, 1)
        y = np.array(values[i*gap: (i+1)*gap]).reshape(-1, 1)
        lr.fit(X, y)

        X_ts_track = mdate.epoch2num(np.array(X)/1000)
        ax.plot_date(X_ts_track, lr.predict(np.array(X).reshape(-1, 1))[:], marker='o', ms=1, c='red')

    ax.set_xlabel('Epoch time')
    ax.set_ylabel('Rate limit track')


def read_data(filedata):
    values = map(int, filedata.readline().rstrip().split(':')[2].split(','))
    tss = map(int, filedata.readline().rstrip().split(':')[2].split(','))
    # ts_track = np.array(map(int, data.readline().rstrip().split(':')[2].split(','))) / 1000
    # ts_track = mdate.epoch2num(ts_track)
    return values, tss

if __name__ == '__main__':
    filepath = os.path.join(file_loc, 'maximum_invert_ts.log')

    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    axs = [ax1, ax2, ax3]
    with open(filepath, 'r') as filedata:
        cnt = 0
        s = 0
        while cnt < 3:
            values, tss = read_data(filedata)
            if s > 9:
                plot_ts_value(values, tss, axs[cnt])
                cnt += 1
            s += 1

    plt.show()
