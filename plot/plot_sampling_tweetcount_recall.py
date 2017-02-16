#!/usr/bin/python

# Usage example:
# python plot_sampling_tweetcount_recall.py

import os
import numpy as np
from scipy import stats
import math
from collections import defaultdict
import matplotlib.pyplot as plt


BASE_DIR = '../'


def get_recall_at_tweetcount(path1, path2):
    tweetcount_vid_dict1 = defaultdict(list)
    with open(path1, 'r') as f1:
        for line in f1:
            vid, tweetcount = line.rstrip().split()
            tweetcount_vid_dict1[int(tweetcount)].append(vid)

    tweetcount_vid_dict2 = defaultdict(list)
    with open(path2, 'r') as f2:
        for line in f2:
            vid, tweetcount = line.rstrip().split()
            tweetcount_vid_dict2[int(tweetcount)].append(vid)

    recalls = []
    for i in xrange(start, end + 1, jump):
        video_set1 = set([item for j in tweetcount_vid_dict1.keys() if j >= i for item in tweetcount_vid_dict1[j]])
        video_set2 = set([item for j in tweetcount_vid_dict2.keys() if j >= i for item in tweetcount_vid_dict2[j]])
        recalls.append(1.0 * len(video_set1.intersection(video_set2)) / len(video_set1))

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

    ax1.errorbar(np.arange(start, end + 1, jump), mean_list, yerr=error_list, c=color, fmt='o-', markersize='2', label=label_text)


if __name__ == '__main__':

    fig, ax1 = plt.subplots(1, 1)
    start = 1
    end = 100
    jump = 1

    complete_25m = os.path.join(BASE_DIR, 'data/complete_tweetcount.txt')
    sample_25m = os.path.join(BASE_DIR, 'data/sample_tweetcount.txt')
    recall_25m = get_recall_at_tweetcount(complete_25m, sample_25m)
    ax1.plot(np.arange(start, end + 1, jump), recall_25m, c='r', label='firehose and filter streaming 25m')

    recall_10ms = []
    for i in xrange(14):
        complete_10m = os.path.join(BASE_DIR, 'data/complete_tweetcount_10m/complete_tweetcount_10m{0}.txt'.format(i))
        sample_10m = os.path.join(BASE_DIR, 'data/sample_tweetcount_10m/sample_tweetcount_10m{0}.txt'.format(i))
        recalls_10m = get_recall_at_tweetcount(complete_10m, sample_10m)
        recall_10ms.append(recalls_10m)

    plot_errorbar(recall_10ms, 'g', 'firehose and filter streaming 10m')

    recall_5ms = []
    for i in xrange(19):
        complete_5m = os.path.join(BASE_DIR, 'data/complete_tweetcount_5m/complete_tweetcount_5m{0}.txt'.format(i))
        sample_5m = os.path.join(BASE_DIR, 'data/sample_tweetcount_5m/sample_tweetcount_5m{0}.txt'.format(i))
        recalls_5m = get_recall_at_tweetcount(complete_5m, sample_5m)
        recall_5ms.append(recalls_5m)

    plot_errorbar(recall_5ms, 'b', 'firehose and filter streaming   5m')

    ax1.set_ylim(ymin=0)
    ax1.set_ylim(ymax=1)
    ax1.set_xlabel('n')
    ax1.set_ylabel('recall')
    ax1.set_title('Figure 2: recall of videos with n tweets')

    plt.legend(loc='lower right')
    plt.show()
