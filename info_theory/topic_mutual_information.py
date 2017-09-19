#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""calculate mutual information between two different topics, i.e, I(Adele; Music),
by constructing 2x2 oc-occurrence matrix
X        0     1
Y  0    10    15
   1    20    25

I(X;Y) = sum(P(x, y) * log( P(x, y)/P(x)/P(y) ))
"""

from __future__ import division, print_function
import os, sys, time, datetime, itertools, operator
from collections import defaultdict
import numpy as np


def test_case():
    # X            0     1
    # Y    0    1000     1
    #      1     100    10
    # X            0     1
    # Y    0       a     b
    #      1       c     d
    a, b, c, d = 1000, 1, 100, 10
    n = a + b + c + d

    p_x0 = (a+c)/n
    p_x1 = (b+d)/n
    p_y0 = (a+b)/n
    p_y1 = (c+d)/n
    p_x0_given_y0 = a/(a+b)
    p_x1_given_y0 = b/(a+b)
    p_x0_given_y1 = c/(c+d)
    p_x1_given_y1 = d/(c+d)
    p_y0_given_x0 = a/(a+c)
    p_y1_given_x0 = c/(a+c)
    p_y0_given_x1 = b/(b+d)
    p_y1_given_x1 = d/(b+d)

    H_X = -(p_x0 * np.log2(p_x0) + p_x1 * np.log2(p_x1))
    H_Y = -(p_y0 * np.log2(p_y0) + p_y1 * np.log2(p_y1))
    H_X_given_Y = -(p_y0 * (p_x0_given_y0 * np.log2(p_x0_given_y0) + p_x1_given_y0 * np.log2(p_x1_given_y0)) +
                    p_y1 * (p_x0_given_y1 * np.log2(p_x0_given_y1) + p_x1_given_y1 * np.log2(p_x1_given_y1)))
    H_Y_given_X = -(p_x0 * (p_y0_given_x0 * np.log2(p_y0_given_x0) + p_y1_given_x0 * np.log2(p_y1_given_x0)) +
                    p_x1 * (p_y0_given_x1 * np.log2(p_y0_given_x1) + p_y1_given_x1 * np.log2(p_y1_given_x1)))

    I_X_Y1 = H_X - H_X_given_Y
    I_X_Y2 = H_Y - H_Y_given_X

    H_X_Y1 = H_X + H_Y_given_X
    H_X_Y2 = H_Y + H_X_given_Y

    # print(I_X_Y1, I_X_Y2, H_X_Y1, H_X_Y2)
    epison = 10**-10
    assert abs(I_X_Y1-I_X_Y2) < epison, 'mutual information does not match'
    assert abs(H_X_Y1-H_X_Y2) < epison, 'joint entropy does not match'
    print('>>> Pass test case!')


def safe_log2(x):
    if x == 0:
        return 0
    else:
        return np.log2(x)


def get_stats(t1, t2):
    d = topic_matrix[t1][t2]
    c = topic_appearance[t2] - d
    b = topic_appearance[t1] - d
    a = overall_cnt - b - c - d
    return a, b, c, d


def get_mutual_information(a, b, c, d):
    n = a + b + c + d

    p_x0 = (a + c) / n
    p_x1 = (b + d) / n
    p_y0 = (a + b) / n
    p_y1 = (c + d) / n
    p_x0_given_y0 = a / (a + b)
    p_x1_given_y0 = b / (a + b)
    p_x0_given_y1 = c / (c + d)
    p_x1_given_y1 = d / (c + d)
    p_y0_given_x0 = a / (a + c)
    p_y1_given_x0 = c / (a + c)
    p_y0_given_x1 = b / (b + d)
    p_y1_given_x1 = d / (b + d)

    H_X = -(p_x0 * safe_log2(p_x0) + p_x1 * safe_log2(p_x1))
    H_Y = -(p_y0 * safe_log2(p_y0) + p_y1 * safe_log2(p_y1))
    H_X_given_Y = -(p_y0 * (p_x0_given_y0 * safe_log2(p_x0_given_y0) + p_x1_given_y0 * safe_log2(p_x1_given_y0)) +
                    p_y1 * (p_x0_given_y1 * safe_log2(p_x0_given_y1) + p_x1_given_y1 * safe_log2(p_x1_given_y1)))
    H_Y_given_X = -(p_x0 * (p_y0_given_x0 * safe_log2(p_y0_given_x0) + p_y1_given_x0 * safe_log2(p_y1_given_x0)) +
                    p_x1 * (p_y0_given_x1 * safe_log2(p_y0_given_x1) + p_y1_given_x1 * safe_log2(p_y1_given_x1)))

    epison = 10**-10
    assert abs(H_X - H_X_given_Y - H_Y + H_Y_given_X) < epison, 'mutual information does not match'

    I_X_Y = H_X - H_X_given_Y
    return I_X_Y


def _load_data(filepath):
    cnt = 0
    with open(filepath, 'r') as fin:
        fin.readline()
        for line in fin:
            topics = line.rstrip().split('\t')[7]
            if not topics == '' and topics != 'NA':
                cnt += 1
                topics = topics.split(',')
                if len(topics) == 1:
                    topic_appearance[topics[0]] += 1
                else:
                    for topic in topics:
                        topic_appearance[topic] += 1
                    for pair in itertools.combinations(topics, r=2):
                        if pair[0] not in topic_matrix:
                            topic_matrix[pair[0]] = defaultdict(int)
                        topic_matrix[pair[0]][pair[1]] += 1
                        if pair[1] not in topic_matrix:
                            topic_matrix[pair[1]] = defaultdict(int)
                        topic_matrix[pair[1]][pair[0]] += 1
    print('>>> Finish loading file {0}!'.format(filepath))
    return cnt


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    start_time = time.time()

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    data_loc = '../../production_data/tweeted_dataset_norm/'
    # {'': (), ...}
    topic_appearance = defaultdict(int)
    topic_matrix = {}
    overall_cnt = 0
    print('>>> Start to load all tweeted dataset...')
    for subdir, _, files in os.walk(data_loc):
        for f in files:
            overall_cnt += _load_data(os.path.join(subdir, f))
    print('>>> Finish loading all data!\n')

    print('overall videos number: {0}'.format(overall_cnt))
    print('number of topics: {0}'.format(len(topic_appearance)))

    # == == == == == == == == Part 3: Load Freebase dictionary == == == == == == == == #
    freebase_loc = '../../production_data/freebase_mid_name.txt'
    freebase_dict = {}
    with open(freebase_loc, 'r') as fin:
        for line in fin:
            mid, entity = line.rstrip().split('\t', 1)
            freebase_dict[mid] = entity

    # == == == == == == == == Part 4: Calculate mutual information for all topic pairs == == == == == == == == #
    # filter with appearance over 100
    frequent_topics = []
    for topic in topic_appearance:
        if topic_appearance[topic] > 100:
            frequent_topics.append(topic)

    topics_mi_dict = {}
    for t1, t2 in itertools.combinations(frequent_topics, r=2):
        d = topic_matrix[t1][t2]
        if d > 100:
            a, b, c, _ = get_stats(t1, t2)
            mi = get_mutual_information(a, b, c, d)
            topics_mi_dict['{0}-{1}'.format(t1, t2)] = mi

    # display 25 largest pairs of topics
    sorted_mi = sorted(topics_mi_dict.items(), key=operator.itemgetter(1), reverse=True)[:1000]
    for item in sorted_mi:
        t1, t2 = item[0].split('-')
        print('>>> Mutual information for {0} and {1}: {2:.4f}'.format(freebase_dict[t1], freebase_dict[t2], item[1]))
        print('{0}, {1}, {2}, {3}'.format(*get_stats(t1, t2)))
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
