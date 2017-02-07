#!/usr/bin/python
# -*- coding: utf-8 -*-

# Usage example:
# python plot_recomnet_prob_dist.py -v e-ORhEE9VVg

import os
import json
import argparse
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


def plot_dist(relevant_list_path, sidebar_list_path):
    encode_dict = {}
    stat_dict1 = defaultdict(int)
    stat_dict2 = defaultdict(int)
    cnt = 1
    with open(relevant_list_path, 'r') as f1:
        for line in f1:
            vid = line.rstrip()
            encode_dict[cnt] = vid
            stat_dict1[vid] = 0
            stat_dict2[vid] = 0
            cnt += 1

    x_axis = np.arange(1, cnt)

    cnt2 = 0
    with open(sidebar_list_path, 'r') as f2:
        for line in f2:
            sidebar_list = line.rstrip().split()
            n = len(sidebar_list)
            seg = n-20 if n>20 else n
            # first layer exposure
            for i in xrange(seg):
                vid = sidebar_list[i]
                if vid in stat_dict1:
                    stat_dict1[vid] += 1
            # second layer exposure
            if n > 20:
                for i in xrange(n-20, n):
                    vid = sidebar_list[i]
                    if vid in stat_dict2:
                        stat_dict2[vid] += 1
            cnt2 += 1

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)

    y_axis1 = [100.0*stat_dict1[encode_dict[cnt]]/cnt2 for cnt in x_axis]
    ax1.scatter(x_axis, y_axis1)

    y_axis2 = [100.0*stat_dict2[encode_dict[cnt]]/cnt2 for cnt in x_axis]
    ax2.scatter(x_axis, y_axis2)

    ax1.set_xlim(xmin=0)
    ax1.set_xlim(xmax=100)
    ax1.set_ylim(ymin=-5)
    ax1.set_ylim(ymax=105)

    plt.show()


if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', help='request video id', required=True)
    args = parser.parse_args()

    relevant_list_path = '../log/{0}_relevant_list.txt'.format(args.v)
    sidebar_list_path = '../log/{0}_sidebar_list.txt'.format(args.v)
    plot_dist(relevant_list_path, sidebar_list_path)