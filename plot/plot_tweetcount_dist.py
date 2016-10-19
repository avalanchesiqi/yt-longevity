# import matplotlib
# matplotlib.use('Agg')

import sys
import numpy as np
from collections import defaultdict
import powerlaw

import matplotlib.pyplot as plt


def plot_basics(filepath, ax, c):
    tweetcounts = np.genfromtxt(filepath, usecols=(1,), comments=None, dtype=np.int32, unpack=True)

    ###
    x, y = powerlaw.pdf(tweetcounts, linear_bins=True)
    ind = y > 0
    y = y[ind]
    x = x[:-1]
    x = x[ind]
    ax.scatter(x, y, color=c, s=.5)
    # ax.set_xscale('log')
    ax.set_yscale('log')
    # powerlaw.plot_pdf(tweetcounts, ax=ax, color='b', linewidth=2)
    # ###
    #
    # from mpl_toolkits.axes_grid.inset_locator import inset_axes
    # ax1in = inset_axes(ax, width="30%", height="30%", loc=3)
    # ax1in.hist(tweetcounts, density=True, color='b')
    # ax1in.set_xticks([])
    # ax1in.set_yticks([])
    #
    # ###
    # fit = powerlaw.Fit(tweetcounts, xmin=1, discrete=True)
    # fit.power_law.plot_pdf(ax=ax, linestyle=':', color='g')
    #
    # ####
    # fit = powerlaw.Fit(tweetcounts, discrete=True)
    # fit.power_law.plot_pdf(ax=ax, linestyle='--', color='g')
    #
    # ax.set_xlabel(r"Tweetcount")
    # ax.set_ylabel(u"p(X)")
    #
    # alpha = fit.power_law.alpha
    # xmin = fit.power_law.xmin
    # textstr = 'alpha: {0:.4f}\nxmin: {1}\nperiod: {2}'.format(alpha, xmin, text)
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # ax.text(0.5, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    # ####

    return


def plot_main():
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig, ax1 = plt.subplots(1, 1)
    r10 = '../../data/test_sampling/2015/10/vid_tweetcount.txt'
    plot_basics(r10, ax1, "r")
    # r20 = '../../data/test_sampling/2015/20/vid_tweetcount.txt'
    # plot_basics(r20, ax1, "Ratio at 20")
    # r30 = '../../data/test_sampling/2015/30/vid_tweetcount.txt'
    # plot_basics(r30, ax1, "Ratio at 30")
    r50 = '../../data/test_sampling/2015/50/vid_tweetcount.txt'
    plot_basics(r50, ax1, "g")
    r80 = '../../data/test_sampling/2015/80/vid_tweetcount.txt'
    plot_basics(r80, ax1, "b")
    # jun_15 = '../../data/full_jun_data/2015/vid_tweetcount.txt'
    # plot_basics(jun_15, ax2, "June' 2015")
    # jun_16 = '../../data/full_jun_data/2016/vid_tweetcount.txt'
    # plot_basics(jun_16, ax3, "June' 2016")

    plt.show()


if __name__ == '__main__':
    plot_main()
