#!/usr/bin/python

# Usage example:
# python plot_sampling_recall_topn.py --file1='<file1>' --file2='<file2>'

import os
import argparse
import operator
from collections import defaultdict
from scipy import stats
import math
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = '../'


def get_recall(path1, path2):
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

    recalls = []
    for i in xrange(start, end + 1, jump):
        video_set1 = set(video_list1[:i])
        video_set2 = set(video_list2[:i])
        recalls.append(1.0*len(video_set1.intersection(video_set2))/i)

    return recalls


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
    parser.add_argument('-f1', '--file1', help='firehose path, relative to base dir', required=True)
    parser.add_argument('-f2', '--file2', help='streaming path, relative to base dir', required=True)
    parser.add_argument('-d', '--dir', help='directory path, relative to base dir')
    args = parser.parse_args()

    fig, ax1 = plt.subplots(1, 1)
    start = 10
    end = 1000
    jump = 10
    # simulate and filter streaming
    recall_list1 = []
    # simulate and firehose
    recall_list2 = []

    file1_path = os.path.join(BASE_DIR, args.file1)

    file2_path = os.path.join(BASE_DIR, args.file2)
    recalls = get_recall(file1_path, file2_path)
    ax1.plot(np.arange(start, end + 1, jump), recalls, c='r', label='firehose and filter streaming')

    dir_path = os.path.join(BASE_DIR, args.dir)
    for subdir, _, files in os.walk(dir_path):
        for f in files:
            recall_list2.append(get_recall(file1_path, os.path.join(subdir, f)))

    plot_errorbar(recall_list2, 'b', 'firehose and simulated data')

    ax1.set_ylim(ymin=0)
    ax1.set_xlabel('n')
    ax1.set_ylabel('recall')
    ax1.set_title('Figure 1: recall of top n tweeted videos')

    plt.legend(loc='best')
    plt.show()
