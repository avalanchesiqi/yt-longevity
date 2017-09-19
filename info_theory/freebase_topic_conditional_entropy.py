#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Predict relative engagement from topic features, with ridge regression and sparse matrix."""

from __future__ import division, print_function
import os, sys
import time, datetime
import numpy as np


def _load_data(filepath, threshold=0.5):
    topic_appearance = {}
    cnt = 0
    with open(filepath, 'r') as fin:
        fin.readline()
        for line in fin:
            _, _, _, _, _, _, _, topics, _, _, _, _, re30, _ = line.rstrip().split('\t', 13)
            if not topics == '' and topics != 'NA':
                cnt += 1
                label = float(re30) >= threshold
                for topic in topics.split(','):
                    if topic in topic_appearance:
                        a, b = topic_appearance[topic]
                    else:
                        a, b = 0, 0
                    if label:
                        topic_appearance[topic] = (a+1, b)
                    else:
                        topic_appearance[topic] = (a, b+1)
    print('>>> Finish loading file {0}!'.format(filepath))
    return topic_appearance, cnt


def conditional_update(dict1, dict2):
    for k, v in dict2.items():
        if k in dict1.keys():
            dict1[k] = (dict1[k][0]+dict2[k][0], dict1[k][1]+dict2[k][1])
        else:
            dict1[k] = dict2[k]
    return dict1


def calculate_conditional_entropy(cnt, y1, y2):
    if y1 == 0 or y2 == 0:
        return 0
    else:
        return (y1/cnt)*np.log2((y1+y2)/y1) + (y2/cnt)*np.log2((y1+y2)/y2)


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    start_time = time.time()
    engagement_threshold = 0.5

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    freebase_loc = '../../production_data/freebase_mid_name.txt'
    freebase_dict = {}
    with open(freebase_loc, 'r') as fin:
        for line in fin:
            mid, entity = line.rstrip().split('\t', 1)
            freebase_dict[mid] = entity

    data_loc = '../../production_data/tweeted_dataset_norm/'
    # {'': (), ...}
    overall_topic_appearance = {}
    overall_cnt = 0
    print('>>> Start to load all tweeted dataset...')
    for subdir, _, files in os.walk(data_loc):
        for f in files:
            file_topic_appearance, file_cnt = _load_data(os.path.join(subdir, f), threshold=engagement_threshold)
            overall_topic_appearance = conditional_update(overall_topic_appearance, file_topic_appearance)
            overall_cnt += file_cnt
    print('>>> Finish loading all data!\n')

    # == == == == == == == == Part 3: Calculate conditional entropy for all topics == == == == == == == == #
    conditional_entropy_dict = {}
    for topic in overall_topic_appearance.keys():
        high, low = overall_topic_appearance[topic]
        conditional_entropy_dict[topic] = (calculate_conditional_entropy(overall_cnt, high, low), high+low)

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])

    # == == == == == == == == Part 4: Sort by entropy then frequency == == == == == == == == #
    sorted_entropy = sorted(conditional_entropy_dict.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)

    # write to txt file
    to_write = True
    if to_write:
        print('>>> Prepare to write to text file...')
        print('>>> Number of videos with freebase topics: {0}'.format(len(sorted_entropy)))
        output_path = './output/freebase_topic_entropy_{0}.txt'.format(engagement_threshold)
        with open(output_path, 'w') as fout:
            for mid, entropy_freq in sorted_entropy:
                fout.write('{0}\t{1}\t{2}\n'.format(freebase_dict[mid], entropy_freq[0], entropy_freq[1]))
