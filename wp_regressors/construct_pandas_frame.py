#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Construct pandas dataframe from predictor pickle output.

Example rows
           Vid      True  Duration   Content     Topic    CTopic       CPS        All       CSP
0  KcgjkCDPOco  0.300155  0.300902  0.343142  0.348659  0.378310  0.589099   0.608004  0.608004 
1  oydbUUFZNPQ  0.339141  0.424607  0.392023  1.000000  0.501331  0.431703   0.472517  0.438917
2  RUAKJSxfgW0  0.694815  0.454994  0.473313  0.587380  0.586567  0.582470   0.610346  0.526601
3  U45p1d_zQEs  0.491463  0.486584  0.514190  0.502563  0.508860  0.602810   0.554918  0.623619
4  wjdjztvb9Hc  0.832867  0.541110  0.556331  1.000000  0.501331  0.781969   0.472517  0.472517
"""

from __future__ import division, print_function
import os
import cPickle as pickle
import pandas as pd

if __name__ == '__main__':
    # construct pandas dataframe if not exists
    dataframe_path = './data/predicted_wp_sparse_df.csv'
    if not os.path.exists(dataframe_path):
        prefix_dir = './output/'
        true_dict_path = os.path.join(prefix_dir, 'true_predictor.p')
        duration_predictor_path = os.path.join(prefix_dir, 'duration_predictor.p')
        content_predictor_path = os.path.join(prefix_dir, 'content_predictor.p')
        topic_predictor_path = os.path.join(prefix_dir, 'sparse_topic_predictor.p')
        content_topic_predictor_path = os.path.join(prefix_dir, 'sparse_content_topic_predictor.p')
        channel_predictor_path = os.path.join(prefix_dir, 'cps_predictor.p')
        all_predictor_path = os.path.join(prefix_dir, 'sparse_all_predictor.p')
        per_channel_predictor_path = os.path.join(prefix_dir, 'csp_predictor_5.p')

        # ground-truth values
        true_dict = pickle.load(open(true_dict_path, 'rb'))
        vids = true_dict.keys()

        # duration predictor
        duration_predictor = pickle.load(open(duration_predictor_path, 'rb'))

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
        duration_data_f = pd.DataFrame(duration_predictor.items(), columns=['Vid', 'Duration'])
        content_data_f = pd.DataFrame(content_predictor.items(), columns=['Vid', 'Content'])
        topic_data_f = pd.DataFrame(topic_predictor.items(), columns=['Vid', 'Topic'])
        content_topic_data_f = pd.DataFrame(content_topic_predictor.items(), columns=['Vid', 'CTopic'])
        channel_data_f = pd.DataFrame(channel_predictor.items(), columns=['Vid', 'CPS'])
        all_data_f = pd.DataFrame(all_predictor.items(), columns=['Vid', 'All'])
        per_channel_data_f = pd.DataFrame(per_channel_predictor.items(), columns=['Vid', 'CSP'])
        data_f = true_data_f.merge(duration_data_f, on='Vid').merge(content_data_f, on='Vid')\
            .merge(topic_data_f, on='Vid').merge(content_topic_data_f, on='Vid').merge(channel_data_f, on='Vid')\
            .merge(all_data_f, on='Vid').merge(per_channel_data_f, on='Vid')

        for name in ['True', 'Duration', 'Content', 'Topic', 'CTopic', 'CPS', 'All', 'CSP']:
            data_f[name] = data_f[name].where(data_f[name] < 1, 1)
            data_f[name] = data_f[name].where(data_f[name] > 0, 0)
        data_f.to_csv(dataframe_path, sep='\t')

        print('header:')
        print(data_f.head())
