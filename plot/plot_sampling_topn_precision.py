#!/usr/bin/python

# Usage example:
# python plot_sampling_topn_precision.py

import os
import argparse
import operator
from collections import defaultdict
from scipy import stats
import math
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = '../'


def get_prec(path1, path2):
    video_list1 = []
    with open(path1, 'r') as f1:
        for line in f1:
            vid, tweetcount = line.rstrip().split()
            video_list1.append(vid)

    video_list2 = []
    with open(path2, 'r') as f2:
        for line in f2:
            vid, tweetcount = line.rstrip().split()
            video_list2.append(vid)

    precs = []
    for i in xrange(start, end + 1, jump):
        video_set1 = set(video_list1[:i])
        video_set2 = set(video_list2[:i])
        precs.append(1.0*len(video_set1.intersection(video_set2))/i)

    return precs


def plot_errorbar(taus_list, color, label_text):
    transpose_taus = map(list, zip(*taus_list))
    z_critical = stats.norm.ppf(q=0.95)
    sample_size = len(transpose_taus[0])

    mean_list = []
    error_list = []

    for taus in transpose_taus:
        mean = np.mean(taus)
        std = np.std(taus)
        error = z_critical * (std / math.sqrt(sample_size))
        mean_list.append(mean)
        error_list.append(error)

    ax1.errorbar(np.arange(start, end + 1, jump), mean_list, yerr=error_list, c=color, fmt='o-',
                 markersize='2', label=label_text)


if __name__ == '__main__':
    # Instantiate the parser
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-f1', '--file1', help='firehose path, relative to base dir', required=True)
    # parser.add_argument('-f2', '--file2', help='streaming path, relative to base dir', required=True)
    # parser.add_argument('-d', '--dir', help='directory path, relative to base dir')
    # args = parser.parse_args()

    fig, ax1 = plt.subplots(1, 1)
    start = 10
    end = 2000
    jump = 10
    # # simulate and filter streaming
    # recall_list1 = []
    # # simulate and firehose
    # recall_list2 = []

    complete_25m = os.path.join(BASE_DIR, 'data/complete_tweetcount.txt')
    sample_25m = os.path.join(BASE_DIR, 'data/sample_tweetcount.txt')
    precs_25m = get_prec(complete_25m, sample_25m)
    ax1.plot(np.arange(start, end + 1, jump), precs_25m, c='r', label='firehose and filter streaming 25m')

    prec_10ms = []
    for i in xrange(14):
        complete_10m = os.path.join(BASE_DIR, 'data/complete_tweetcount_10m/complete_tweetcount_10m{0}.txt'.format(i))
        sample_10m = os.path.join(BASE_DIR, 'data/sample_tweetcount_10m/sample_tweetcount_10m{0}.txt'.format(i))
        precs_10m = get_prec(complete_10m, sample_10m)
        prec_10ms.append(precs_10m)

    plot_errorbar(prec_10ms, 'g', 'firehose and filter streaming 10m')
    # ax1.plot(np.arange(start, end + 1, jump), precs_10m, c='g', label='firehose and filter streaming 10m')

    prec_5ms = []
    for i in xrange(19):
        complete_5m = os.path.join(BASE_DIR, 'data/complete_tweetcount_5m/complete_tweetcount_5m{0}.txt'.format(i))
        sample_5m = os.path.join(BASE_DIR, 'data/sample_tweetcount_5m/sample_tweetcount_5m{0}.txt'.format(i))
        precs_5m = get_prec(complete_5m, sample_5m)
        prec_5ms.append(precs_5m)

    plot_errorbar(prec_5ms, 'b', 'firehose and filter streaming   5m')
    # ax1.plot(np.arange(start, end + 1, jump), precs_5m, c='b', label='firehose and filter streaming 5m')

    # ax1.set_xscale('log')
    ax1.set_ylim(ymin=0)
    ax1.set_ylim(ymax=1)
    ax1.set_xlabel('n')
    ax1.set_ylabel('precision')
    ax1.set_title('Figure 1: precision of top n tweeted videos')

    plt.legend(loc='lower right')
    plt.show()
