#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""calculate conditional entropy between topic type and relative engagement, i.e, I(Music; eta),
by constructing 2x20 oc-occurrence matrix
X                0     1
Y     0-0.05    10    15
   0.05-0.10    20    25
       ....
   0.95-1.00    30    35

I(X;Y) = sum(P(x, y) * log( P(x, y)/P(x)/P(y) ))
"""

from __future__ import division, print_function
import os, sys, time, datetime, operator
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def safe_log2(x):
    if x == 0:
        return 0
    else:
        return np.log2(x)


def get_conditional_entropy(topic_eta):
    # calculate condition entropy when topic appears
    binned_topic_eta = {i: 0 for i in range(bin_num)}
    for eta in topic_eta:
        binned_topic_eta[min(int(eta / bin_gap), bin_num - 1)] += 1

    p_Y_given_x1 = [binned_topic_eta[i] / len(topic_eta) for i in range(bin_num)]
    return -np.sum([p * safe_log2(p) for p in p_Y_given_x1])


def _load_data(filepath):
    with open(filepath, 'r') as fin:
        fin.readline()
        for line in fin:
            _, _, _, _, _, _, _, topics, _, _, _, _, re30, _ = line.rstrip().split('\t', 13)
            if not topics == '' and topics != 'NA':
                topics = topics.split(',')
                re30 = float(re30)
                for topic in topics:
                    if topic in mid_type_dict:
                        freebase_types = mid_type_dict[topic].split(',')
                        for ft in freebase_types:
                            if ft != 'common' and ft != 'type_ontology':
                                type_eta_dict[ft].append(re30)
                                type_eta_counter_dict[ft] += 1
    print('>>> Finish loading file {0}!'.format(filepath))
    return


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    start_time = time.time()
    bin_gap = 0.05
    bin_num = int(1 / bin_gap)

    # == == == == == == == == Part 2: Load Freebase dictionary == == == == == == == == #
    freebase_loc = '../../production_data/freebase_mid_type_name.txt'
    mid_type_dict = {}
    with open(freebase_loc, 'r') as fin:
        for line in fin:
            mid, type, _ = line.rstrip().split('\t', 2)
            mid_type_dict[mid] = type

    # == == == == == == == == Part 3: Load dataset == == == == == == == == #
    # data_loc = '../../production_data/tweeted_dataset_norm/'
    data_loc = '../../production_data/sample_tweeted_dataset/'
    type_eta_dict = defaultdict(list)
    type_eta_counter_dict = defaultdict(int)
    print('>>> Start to load all tweeted dataset...')
    for subdir, _, files in os.walk(data_loc):
        for f in files:
            _load_data(os.path.join(subdir, f))
    print('>>> Finish loading all data!\n')
    print('number of types: {0}'.format(len(type_eta_dict)))

    # == == == == == Part 4: Calculate conditional entropy for topic type and relative engagement == == == == == #
    # only count 1000 largest types
    sorted_type_eta_counter = sorted(type_eta_counter_dict.items(), key=operator.itemgetter(1), reverse=True)[:1000]
    print('largest 100 clusters')
    print(sorted_type_eta_counter)

    type_conditional_entropy_dict = {}
    for type, _ in sorted_type_eta_counter:
        # type size, conditional entropy, mean eta value
        type_conditional_entropy_dict[type] = (type_eta_counter_dict[type], get_conditional_entropy(type_eta_dict[type]), np.mean(type_eta_dict[type]))

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])

    # == == == == == == == == Part 5: Generate scatter plots == == == == == == == == #
    to_plot = True
    if to_plot:
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        cornflower_blue = '#6495ed'
        tomato = '#ff6347'

        keys = type_conditional_entropy_dict.keys()
        x_axis = [type_conditional_entropy_dict[x][0] for x in keys]
        y_axis = [type_conditional_entropy_dict[x][1] for x in keys]
        alphas = [type_conditional_entropy_dict[x][2] for x in keys]
        ax1.scatter(x_axis, y_axis, facecolors='none', edgecolors='k')

        ax1.set_xscale('log')
        ax1.set_ylim(ymin=0)
        ax1.set_xlabel('Cluster size', fontsize=16)
        ax1.set_ylabel('Conditional entropy', fontsize=16)

        plt.tight_layout()
        plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
#
#
# def make_colormap(seq):
#     """Return a LinearSegmentedColormap
#     seq: a sequence of floats and RGB-tuples. The floats should be increasing
#     and in the interval (0,1).
#     """
#     seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
#     cdict = {'red': [], 'green': [], 'blue': []}
#     for i, item in enumerate(seq):
#         if isinstance(item, float):
#             r1, g1, b1 = seq[i - 1]
#             r2, g2, b2 = seq[i + 1]
#             cdict['red'].append([item, r1, r2])
#             cdict['green'].append([item, g1, g2])
#             cdict['blue'].append([item, b1, b2])
#     return mcolors.LinearSegmentedColormap('CustomMap', cdict)
#
#
# c = mcolors.ColorConverter().to_rgb
# rvb = make_colormap([(0.3921, 0.5843, 0.9294), c('white'), 0.5, c('white'), (1.0, 0.3882, 0.2784)])
# N = 1000
# array_dg = np.random.uniform(0, 10, size=(N, 2))
# colors = np.random.uniform(0, 1, size=(N,))
# plt.scatter(array_dg[:, 0], array_dg[:, 1], c=colors, cmap=rvb)
# plt.colorbar()
# plt.show()
