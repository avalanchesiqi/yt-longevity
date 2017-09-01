#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Construct pandas dataframe from predictor pickle output.

Example rows
   Unnamed: 0  Vid   True   Content     Topic    CTopic       CPS       All        CSP  
0           0    1  0.481  0.607106  0.609333  0.665838  0.870050  0.876945   0.876945 
1           1    1  0.295  0.507591  0.634284  0.514392  0.422758  0.488751   0.376236 
"""

from __future__ import division, print_function
import os
import cPickle as pickle
import pandas as pd

if __name__ == '__main__':
    # construct pandas dataframe if not exists
    dataframe_path = './data/predicted_sparse_df.csv'
    if not os.path.exists(dataframe_path):
        prefix_dir = './output/'
        true_dict_path = os.path.join(prefix_dir, 'true_predictor.p')
        content_predictor_path = os.path.join(prefix_dir, 'content_predictor.p')
        topic_predictor_path = os.path.join(prefix_dir, 'sparse_topic_predictor.p')
        content_topic_predictor_path = os.path.join(prefix_dir, 'sparse_content_topic_predictor.p')
        channel_predictor_path = os.path.join(prefix_dir, 'cps_predictor.p')
        all_predictor_path = os.path.join(prefix_dir, 'sparse_all_predictor.p')
        per_channel_predictor_path = os.path.join(prefix_dir, 'csp_predictor_5.p')

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

        # generate pandas dataframe
        true_data_f = pd.DataFrame(true_dict.items(), columns=['Vid', 'True'])
        content_data_f = pd.DataFrame(content_predictor.items(), columns=['Vid', 'Content'])
        topic_data_f = pd.DataFrame(topic_predictor.items(), columns=['Vid', 'Topic'])
        content_topic_data_f = pd.DataFrame(content_topic_predictor.items(), columns=['Vid', 'CTopic'])
        channel_data_f = pd.DataFrame(channel_predictor.items(), columns=['Vid', 'CPS'])
        all_data_f = pd.DataFrame(all_predictor.items(), columns=['Vid', 'All'])
        per_channel_data_f = pd.DataFrame(per_channel_predictor.items(), columns=['Vid', 'CSP'])
        data_f = true_data_f.merge(content_data_f, on='Vid').merge(topic_data_f, on='Vid')\
            .merge(content_topic_data_f, on='Vid').merge(channel_data_f, on='Vid')\
            .merge(all_data_f, on='Vid').merge(per_channel_data_f, on='Vid')

        data_f = data_f.where(data_f < 1, 1)
        data_f = data_f.where(data_f > 0, 0)
        data_f.to_csv(dataframe_path, sep='\t')

        print('header:')
        print(data_f.head())
