#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""calculate conditional entropy between topic and relative engagement, i.e, I(Adele; eta),
by constructing 2x20 oc-occurrence matrix
X                0     1
Y     0-0.05    10    15
   0.05-0.10    20    25
       ....
   0.95-1.00    30    35

I(X;Y) = sum(P(x, y) * log( P(x, y)/P(x)/P(y) ))
"""

from __future__ import division, print_function
import os, sys, time, datetime, itertools, operator
from collections import defaultdict
import numpy as np


def safe_log2(x):
    if x == 0:
        return 0
    else:
        return np.log2(x)


def get_conditional_entropy(engagement_col, topic_eta):
    # calculate condition entropy when topic appears
    binned_topic_eta = {i: 0 for i in range(bin_num)}
    for eta in topic_eta:
        binned_topic_eta[min(int(eta / bin_gap), bin_num - 1)] += 1
    n = np.sum(engagement_col)

    p_x1 = len(topic_eta) / n
    p_Y_given_x1 = [binned_topic_eta[i] / len(topic_eta) for i in range(bin_num)]
    return -p_x1 * np.sum([p * safe_log2(p) for p in p_Y_given_x1])


def _load_data(filepath):
    with open(filepath, 'r') as fin:
        fin.readline()
        for line in fin:
            _, _, _, _, _, _, _, topics, _, _, _, _, re30, _ = line.rstrip().split('\t', 13)
            if not topics == '' and topics != 'NA':
                topics = topics.split(',')
                re30 = float(re30)
                engagement_col[min(int(re30/bin_gap), bin_num-1)] += 1
                for topic in topics:
                    if topic not in topic_row:
                        topic_row[topic] = [re30]
                    else:
                        topic_row[topic].append(re30)
    print('>>> Finish loading file {0}!'.format(filepath))
    return


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    start_time = time.time()
    bin_gap = 0.05
    bin_num = int(1 / bin_gap)

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    data_loc = '../../production_data/tweeted_dataset_norm/'
    # data_loc = '../../production_data/sample_tweeted_dataset/'
    engagement_col = [0]*bin_num
    topic_row = {}
    print('>>> Start to load all tweeted dataset...')
    for subdir, _, files in os.walk(data_loc):
        for f in files:
            _load_data(os.path.join(subdir, f))
    print('>>> Finish loading all data!\n')
    print('number of topics: {0}'.format(len(topic_row)))

    # == == == == == == == == Part 3: Load Freebase dictionary == == == == == == == == #
    freebase_loc = '../../production_data/freebase_mid_name.txt'
    freebase_dict = {}
    with open(freebase_loc, 'r') as fin:
        for line in fin:
            mid, entity = line.rstrip().split('\t', 1)
            freebase_dict[mid] = entity

    # == == == == == Part 4: Calculate mutual information for topic and relative engagement == == == == == #
    # filter with appearance over 1000
    frequent_topics = []
    for topic in topic_row:
        if len(topic_row[topic]) > 1000:
            frequent_topics.append(topic)
    print('number of frequent topics: {0}'.format(len(frequent_topics)))

    topic_eta_conditional_entropy_dict = {}
    for topic in frequent_topics:
        ce = get_conditional_entropy(engagement_col, topic_row[topic])
        topic_eta_conditional_entropy_dict[topic] = ce

    # display 100 smallest topics
    sorted_ce = sorted(topic_eta_conditional_entropy_dict.items(), key=operator.itemgetter(1))
    for item in sorted_ce:
        if item[0] in freebase_dict:
            print('>>> Conditional entropy for {2} {0}: {1:.4f} in {3} occurs'.format(freebase_dict[item[0]], item[1], item[0], len(topic_row[item[0]])))
            # print('-'*79)

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])

    binned_topic_eta = {i: 0 for i in range(bin_num)}
    for eta in topic_row['/m/013f39m8']:
        binned_topic_eta[min(int(eta / bin_gap), bin_num - 1)] += 1
    print(binned_topic_eta)

    # == == == == == == == == Part 4: Sort by entropy then frequency == == == == == == == == #
    # sorted_entropy = sorted(conditional_entropy_dict.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)

    # write to txt file
    to_write = False
    if to_write:
        print('>>> Prepare to write to text file...')
        output_path = './output/freebase_topic_entropy_{0}.txt'.format(engagement_threshold)
        with open(output_path, 'w') as fout:
            for mid, entropy_freq in sorted_entropy:
                fout.write('{0}\t{1}\t{2}\n'.format(freebase_dict[mid], entropy_freq[0], entropy_freq[1]))
