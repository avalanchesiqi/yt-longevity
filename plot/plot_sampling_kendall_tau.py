#!/usr/bin/python

# Usage example:
# python plot_sampling_kendall_tau.py --file1='<file1>' --file2='<file2>'
# python plot_sampling_kendall_tau.py --file1=data/sample_tweetcount.txt --file2=simulate_tweetcount_000.txt
# python plot_sampling_kendall_tau.py -f1 data/sample_tweetcount.txt -f2 data/complete_tweetcount.txt -d data/simulate_tweetcount

import os
import argparse
import operator
from collections import defaultdict
from scipy import stats
import math
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = '../'


def get_kendall_tau(path1, path2):
    item_ranking = defaultdict(list)
    with open(path1, 'r') as f:
        for line in f:
            vid, tweetcount = line.rstrip().split()
            item_ranking[vid].append(int(tweetcount))

    with open(path2, 'r') as f:
        for line in f:
            vid, tweetcount = line.rstrip().split()
            if vid not in item_ranking:
                item_ranking[vid].append(0)
            item_ranking[vid].append(int(tweetcount))

    # fill zero in vid not appear in file2
    for tweetcounts in item_ranking.values():
        if len(tweetcounts) == 1:
            tweetcounts.append(0)

    # sort by value of file1
    sorted_item_ranking = sorted(item_ranking.items(), key=operator.itemgetter(1), reverse=True)
    file1_list = []
    file2_list = []
    for item in sorted_item_ranking:
        file1_list.append(item[1][0])
        file2_list.append(item[1][1])

    taus = []
    for i in xrange(start, end + 1, jump):
        tau, p_value = stats.kendalltau(file1_list[:i], file2_list[:i])
        taus.append(tau)

    return taus


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
    parser = argparse.ArgumentParser()
    parser.add_argument('-f1', '--file1', help='file1 path, relative to base dir', required=True)
    parser.add_argument('-f2', '--file2', help='file2 path, relative to base dir')
    parser.add_argument('-d', '--dir', help='directory path, relative to base dir')
    args = parser.parse_args()

    fig, ax1 = plt.subplots(1, 1)
    start = 10
    end = 5000
    jump = 10
    # simulate and filter streaming
    taus_list1 = []
    # simulate and firehose
    taus_list2 = []

    file1_path = os.path.join(BASE_DIR, args.file1)
    if args.file2 is not None:
        file2_path = os.path.join(BASE_DIR, args.file2)
        taus = get_kendall_tau(file2_path, file1_path)
        ax1.plot(np.arange(start, end+1, jump), taus, c='r', label='firehose and filter streaming')
    if args.dir is not None:
        dir_path = os.path.join(BASE_DIR, args.dir)
        for subdir, _, files in os.walk(dir_path):
            for f in files:
                taus_list1.append(get_kendall_tau(file1_path, os.path.join(subdir, f)))
                taus_list2.append(get_kendall_tau(file2_path, os.path.join(subdir, f)))

        # plot_errorbar(taus_list1, 'b', 'filter streaming and simulated data')
        plot_errorbar(taus_list2, 'b', 'firehose and simulated data')

    ax1.set_ylim(ymin=0)
    ax1.set_xlabel('n')
    ax1.set_ylabel('tau')
    ax1.set_title('Figure 1: Kendall tau of most tweeted videos')

    plt.legend(loc='best')
    plt.show()
