#!/usr/bin/python

# Usage example:
# plot_sampling_recall_tweetcount.py -f1 data/complete_tweetcount.txt -f2 data/sample_tweetcount.txt -d data/simulate_tweetcount

import os
import numpy as np
import argparse
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

    recalls1 = []
    for i in xrange(start, end + 1, jump):
        video_set1 = set([item for j in tweetcount_vid_dict1.keys() if j >= i for item in tweetcount_vid_dict1[j]])
        video_set2 = set([item for j in tweetcount_vid_dict2.keys() for item in tweetcount_vid_dict2[j]])
        recalls1.append(1.0*len(video_set1.intersection(video_set2))/len(video_set1))

    recalls2 = []
    for i in xrange(start, end + 1, jump):
        video_set1 = set([item for j in tweetcount_vid_dict1.keys() for item in tweetcount_vid_dict1[j]])
        video_set2 = set([item for j in tweetcount_vid_dict2.keys() if j >= i for item in tweetcount_vid_dict2[j]])
        recalls2.append(1.0 * len(video_set1.intersection(video_set2)) / len(video_set1))

    recalls3 = []
    for i in xrange(start, end + 1, jump):
        video_set1 = set([item for j in tweetcount_vid_dict1.keys() if j >= i for item in tweetcount_vid_dict1[j]])
        video_set2 = set([item for j in tweetcount_vid_dict2.keys() if j >= i for item in tweetcount_vid_dict2[j]])
        recalls3.append(1.0 * len(video_set1.intersection(video_set2)) / len(video_set1))

    return recalls1, recalls2, recalls3


if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-f1', '--file1', help='firehose path, relative to base dir', required=True)
    parser.add_argument('-f2', '--file2', help='streaming path, relative to base dir', required=True)
    parser.add_argument('-d', '--dir', help='directory path, relative to base dir')
    args = parser.parse_args()

    fig, ax1 = plt.subplots(1, 1)
    start = 1
    end = 100
    jump = 1
    # firehose and simulate
    recall_list = []

    file1_path = os.path.join(BASE_DIR, args.file1)

    file2_path = os.path.join(BASE_DIR, args.file2)
    recalls1, recalls2, recalls3 = get_recall_at_tweetcount(file1_path, file2_path)
    ax1.plot(np.arange(start, end + 1, jump), recalls1, c='r', label='recall of videos with n tweets latent')
    ax1.plot(np.arange(start, end + 1, jump), recalls2, c='b', label='recall of videos with n tweets observed')
    ax1.plot(np.arange(start, end + 1, jump), recalls3, c='g', label='recall of videos with n tweets observed and latent')

    ax1.set_ylim(ymin=-0.03)
    ax1.set_ylim(ymax=1.03)
    ax1.set_xlabel('n')
    ax1.set_ylabel('recall')
    ax1.set_title('Figure 1: recall of videos with n tweets')

    plt.legend(loc='lower right')
    plt.show()
