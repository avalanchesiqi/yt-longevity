#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Construct pandas dataframe from predictor pickle output.

Example rows
           Vid   True   Content     Topic    CTopic       CPS       All       CSP   Duration
0  KcgjkCDPOco  0.498  0.608499  0.610826  0.667708  0.871146  0.878522  0.878522       1212
1  oydbUUFZNPQ  0.301  0.405562  0.635107  0.515533  0.424186  0.489899  0.435074        350
2  RUAKJSxfgW0  0.945  0.462427  0.737947  0.738719  0.699488  0.715835  0.450224        254
3  U45p1d_zQEs  0.512  0.504788  0.491619  0.501147  0.127934  0.142407  0.000000        209
4  wjdjztvb9Hc  0.988  0.523769  0.635107  0.515533  0.994331  0.489899  0.489899        160
"""

from __future__ import division, print_function
import os
import cPickle as pickle
import pandas as pd

if __name__ == '__main__':
    # construct pandas dataframe if not exists
    dataframe_path = './data/predicted_re_sparse_df.csv'
    if not os.path.exists(dataframe_path):
        prefix_dir = './output/'
        true_dict_path = os.path.join(prefix_dir, 'true_predictor.p')
        content_predictor_path = os.path.join(prefix_dir, 'content_predictor.p')
        topic_predictor_path = os.path.join(prefix_dir, 'sparse_topic_predictor.p')
        content_topic_predictor_path = os.path.join(prefix_dir, 'sparse_content_topic_predictor.p')
        channel_predictor_path = os.path.join(prefix_dir, 'cps_predictor.p')
        all_predictor_path = os.path.join(prefix_dir, 'sparse_all_predictor.p')
        per_channel_predictor_path = os.path.join(prefix_dir, 'csp_predictor_5.p')
        test_duration_path = os.path.join(prefix_dir, 'test_duration.p')

        # ground-truth values
        true_dict = pickle.load(open(true_dict_path, 'rb'))
        vids = true_dict.keys()

        # content predictor
        content_predictor = pickle.load(open(content_predictor_path, 'rb'))

        # topic predictor
        topic_predictor = pickle.load(open(topic_predictor_path, 'rb'))
        for vid in vids:
            if vid not in topic_predictor:
                topic_predictor[vid] = 0.5

        # content topic predictor
        content_topic_predictor = pickle.load(open(content_topic_predictor_path, 'rb'))
        for vid in vids:
            if vid not in content_topic_predictor:
                content_topic_predictor[vid] = content_predictor[vid]

        # channel past success predictor
        channel_predictor = pickle.load(open(channel_predictor_path, 'rb'))
        for vid in vids:
            if vid not in channel_predictor:
                channel_predictor[vid] = 0.5

        # all features predictor
        all_predictor = pickle.load(open(all_predictor_path, 'rb'))
        for vid in vids:
            if vid not in all_predictor:
                if channel_predictor[vid] != 0.5:
                    all_predictor[vid] = channel_predictor[vid]
                else:
                    all_predictor[vid] = content_topic_predictor[vid]

        # per channel predictor
        per_channel_predictor = pickle.load(open(per_channel_predictor_path, 'rb'))
        for vid in vids:
            if vid not in per_channel_predictor:
                per_channel_predictor[vid] = all_predictor[vid]

        # test duration
        test_duration = pickle.load(open(test_duration_path, 'rb'))

        # generate pandas dataframe
        true_data_f = pd.DataFrame(true_dict.items(), columns=['Vid', 'True'])
        content_data_f = pd.DataFrame(content_predictor.items(), columns=['Vid', 'Content'])
        topic_data_f = pd.DataFrame(topic_predictor.items(), columns=['Vid', 'Topic'])
        content_topic_data_f = pd.DataFrame(content_topic_predictor.items(), columns=['Vid', 'CTopic'])
        channel_data_f = pd.DataFrame(channel_predictor.items(), columns=['Vid', 'CPS'])
        all_data_f = pd.DataFrame(all_predictor.items(), columns=['Vid', 'All'])
        per_channel_data_f = pd.DataFrame(per_channel_predictor.items(), columns=['Vid', 'CSP'])
        test_duration_data_f = pd.DataFrame(test_duration.items(), columns=['Vid', 'Duration'])
        data_f = true_data_f.merge(content_data_f, on='Vid').merge(topic_data_f, on='Vid')\
            .merge(content_topic_data_f, on='Vid').merge(channel_data_f, on='Vid')\
            .merge(all_data_f, on='Vid').merge(per_channel_data_f, on='Vid')\
            .merge(test_duration_data_f, on='Vid')

        for name in ['True', 'Content', 'Topic', 'CTopic', 'CPS', 'All', 'CSP']:
            data_f[name] = data_f[name].where(data_f[name] < 1, 1)
            data_f[name] = data_f[name].where(data_f[name] > 0, 0)
        data_f.to_csv(dataframe_path, sep='\t')

        print('header:')
        print(data_f.head())
