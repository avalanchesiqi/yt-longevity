# import matplotlib
# matplotlib.use('Agg')

import sys
import numpy as np
from collections import defaultdict
import powerlaw

import matplotlib.pyplot as plt


def plot_basics(filepath, ax, text):
    tweetcounts = np.genfromtxt(filepath, usecols=(1,), comments=None, dtype=np.int32, unpack=True)
    n = len(tweetcounts)

    tc_freq_dict = defaultdict(int)
    for tc in tweetcounts:
        tc_freq_dict[tc] += 1

    sorted_x = sorted(tc_freq_dict.keys())
    x = [i for i in sorted_x]
    y = [1.0*tc_freq_dict[i]/n for i in x]

    ####
    powerlaw.plot_pdf(tweetcounts, ax=ax, color='b', linewidth=2)

    ax.scatter(x, y, s=1, color='r')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(ymin=10 ** -7)
    ax.set_xlabel(r"Tweetcount")
    ax.set_ylabel(u"p(X)")

    fit = powerlaw.Fit(tweetcounts)
    alpha = fit.power_law.alpha
    xmin = fit.power_law.xmin

    textstr = 'alpha: {0}\nxmin: {1}\nperiod: {2}'.format(alpha, xmin, text)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.5, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ####

    return


def plot_main():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    jun_14 = '../../data/full_jun_data/2014/vid_tweetcount.txt'
    plot_basics(jun_14, ax1, "June' 2014")
    jun_15 = '../../data/full_jun_data/2015/vid_tweetcount.txt'
    plot_basics(jun_15, ax2, "June' 2015")
    jun_16 = '../../data/full_jun_data/2016/vid_tweetcount.txt'
    plot_basics(jun_16, ax3, "June' 2016")

    plt.show()


if __name__ == '__main__':
    plot_main()
