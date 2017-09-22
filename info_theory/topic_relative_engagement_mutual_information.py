#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""calculate mutual information between topic and relative engagement, i.e, I(Adele; eta),
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


def test_case():
    adele = [0.23, 0.45, 0.67, 0.56, 0.87, 0.68]
    overall = [50]*bin_num
    binned_adele = {i: 0 for i in range(bin_num)}
    for eta in adele:
        binned_adele[min(int(eta/bin_gap), bin_num-1)] += 1
    n = np.sum(overall)

    p_X = [1-len(adele)/n, len(adele)/n]
    p_Y = [y/n for y in overall]

    p_Y_given_X = [[(y-binned_adele[i])/(n-len(adele)) for i, y in enumerate(overall)],
                   [binned_adele[i] / len(adele) for i in range(bin_num)]]
    p_X_given_Y = [[(overall[i]-binned_adele[i])/overall[i], binned_adele[i]/overall[i]] for i in range(len(p_Y))]

    H_X = -np.sum([p_x * safe_log2(p_x) for p_x in p_X])
    H_Y = -np.sum([p_y * safe_log2(p_y) for p_y in p_Y])

    H_X_given_Y = -np.sum([p_Y[i] * np.sum([p * safe_log2(p) for p in p_X_given_Y[i]]) for i in range(len(p_Y))])
    H_Y_given_X = -np.sum([p_X[i] * np.sum([p * safe_log2(p) for p in p_Y_given_X[i]]) for i in range(len(p_X))])

    I_X_Y1 = H_X - H_X_given_Y
    I_X_Y2 = H_Y - H_Y_given_X

    H_X_Y1 = H_X + H_Y_given_X
    H_X_Y2 = H_Y + H_X_given_Y

    epison = 10**-10
    assert abs(I_X_Y1-I_X_Y2) < epison, 'mutual information does not match'
    assert abs(H_X_Y1-H_X_Y2) < epison, 'joint entropy does not match'
    print('>>> Pass test case!')


def safe_log2(x):
    if x == 0:
        return 0
    else:
        return np.log2(x)


def get_mutual_information(engagement_col, topic_eta):
    binned_topic_eta = {i: 0 for i in range(bin_num)}
    for eta in topic_eta:
        binned_topic_eta[min(int(eta / bin_gap), bin_num - 1)] += 1
    n = np.sum(engagement_col)

    p_X = [1 - len(topic_eta) / n, len(topic_eta) / n]
    p_Y = [y / n for y in engagement_col]

    p_Y_given_X = [[(y - binned_topic_eta[i]) / (n - len(topic_eta)) for i, y in enumerate(engagement_col)],
                   [binned_topic_eta[i] / len(topic_eta) for i in range(bin_num)]]
    p_X_given_Y = [[(engagement_col[i] - binned_topic_eta[i]) / engagement_col[i], binned_topic_eta[i] / engagement_col[i]] for i in range(len(p_Y))]

    H_X = -np.sum([p_x * safe_log2(p_x) for p_x in p_X])
    H_Y = -np.sum([p_y * safe_log2(p_y) for p_y in p_Y])

    H_X_given_Y = -np.sum([p_Y[i] * np.sum([p * safe_log2(p) for p in p_X_given_Y[i]]) for i in range(len(p_Y))])
    H_Y_given_X = -np.sum([p_X[i] * np.sum([p * safe_log2(p) for p in p_Y_given_X[i]]) for i in range(len(p_X))])

    I_X_Y1 = H_X - H_X_given_Y
    I_X_Y2 = H_Y - H_Y_given_X

    epison = 10 ** -10
    assert abs(I_X_Y1 - I_X_Y2) < epison, 'mutual information does not match'
    return I_X_Y1


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
    # filter with appearance over 100
    frequent_topics = []
    for topic in topic_row:
        if len(topic_row[topic]) > 100:
            frequent_topics.append(topic)

    topic_eta_mi_dict = {}
    for topic in frequent_topics:
        mi = get_mutual_information(engagement_col, topic_row[topic])
        topic_eta_mi_dict[topic] = mi

    # display 100 largest pairs of topics
    sorted_mi = sorted(topic_eta_mi_dict.items(), key=operator.itemgetter(1), reverse=True)[:100]
    for item in sorted_mi:
        if item[0] in freebase_dict:
            print('>>> Mutual information for {2} {0}: {1:.4f}'.format(freebase_dict[item[0]], item[1], item[0]))

            if freebase_dict[item[0]] == 'Minecraft':
                binned_topic_eta = {i: 0 for i in range(bin_num)}
                for eta in topic_row[item[0]]:
                    binned_topic_eta[min(int(eta / bin_gap), bin_num - 1)] += 1
                print(binned_topic_eta)
            print('-'*79)

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])

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
