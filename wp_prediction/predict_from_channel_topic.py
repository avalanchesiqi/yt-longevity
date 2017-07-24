#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os
import pandas as pd
import numpy as np
import cPickle as pickle
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import Ridge


def get_wp(duration, percentile):
    bin_idx = np.sum(duration_split_points < duration)
    duration_bin = dur_engage_map[bin_idx]
    percentile = int(round(percentile * 1000))
    wp_percentile = duration_bin[percentile]
    return wp_percentile


def get_wp_list(duration_list, percentile_list):
    wp_list = []
    for d, p in zip(duration_list, percentile_list):
        wp_list.append(get_wp(d, p))
    return wp_list


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    dur_engage_str_map = pickle.load(open('dur_engage_map.p', 'rb'))
    dur_engage_map = {key: list(map(float, value.split(','))) for key, value in dur_engage_str_map.items()}

    duration_split_points = np.array(dur_engage_map['duration'])

    train_channel_cnt_map = pickle.load(open('norm_predict_results/train_channel_cnt.p', 'rb'))
    test_channel_cnt_map = pickle.load(open('norm_predict_results/test_channel_cnt.p', 'rb'))

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    predict_result_dict = {}

    train_loc = '../../data/production_data/random_channel/train_data'
    test_loc = '../../data/production_data/random_channel/test_data'

    for subdir, _, files in os.walk(test_loc):
        for f in files:
            test_df = pd.read_csv(os.path.join(subdir, f), sep='\t', header=None,
                                  names=['vid', 'duration', 'definition', 'category', 'lang', 'channel', 'topics',
                                         'total_view', 'true_wp', 'wp_percentile'],
                                  dtype={'duration': int, 'definition': int, 'category': int,
                                         'wp_percentile': float})
            # if we have observed this channel before
            if f in train_channel_cnt_map and train_channel_cnt_map[f] > 4:
                sub_f = f[:4]
                train_data_path = os.path.join(train_loc, sub_f, f)
                # get past success
                train_df = pd.read_csv(train_data_path, sep='\t', header=None,
                                       names=['vid', 'duration', 'definition', 'category', 'lang', 'channel', 'topics',
                                              'total_view', 'true_wp', 'wp_percentile'],
                                       dtype={'duration': int, 'definition': int, 'category': int,
                                              'wp_percentile': float})
                train_num = train_df.shape[0]
                data_df = train_df.append(test_df)

                train_df = data_df[:train_num]
                test_df = data_df[train_num:]

                topic_cnt = 0
                topic_dict = {}
                for x in data_df['topics'].values.tolist():
                    for topic in x.split(','):
                        if topic not in topic_dict:
                            topic_dict[topic] = topic_cnt
                            topic_cnt += 1

                train_matrix = []
                with open(train_data_path, 'r') as fin1:
                    for train_line in fin1:
                        row = np.zeros(topic_cnt + 2)
                        _, duration, _, _, _, _, topics, _, _, watch_percentile = train_line.rstrip().split('\t')
                        topics = topics.split(',')
                        for topic in topics:
                            row[topic_dict[topic]] = 1
                        # row[-2] = np.log10(int(duration))
                        row[-2] = int(duration)
                        row[-1] = float(watch_percentile)
                        train_matrix.append(row)
                train_matrix = np.array(train_matrix)
                train_x = train_matrix[:, :-1]
                train_y = train_matrix[:, -1].reshape(-1, 1)

                ridge_model = Ridge(fit_intercept=True)
                ridge_model.fit(train_x, train_y)

                test_x = []
                test_vids = []
                test_matrix = []
                with open(os.path.join(subdir, f), 'r') as fin2:
                    for test_line in fin2:
                        row = np.zeros(topic_cnt + 1)
                        vid, duration, _, _, _, _, topics, _, _, _ = test_line.rstrip().split('\t')
                        topics = topics.split(',')
                        for topic in topics:
                            row[topic_dict[topic]] = 1
                        # row[-1] = np.log10(int(duration))
                        row[-1] = int(duration)
                        test_vids.append(vid)
                        test_matrix.append(row)
                test_matrix = np.array(test_matrix)

                pred_test_y = ridge_model.predict(test_matrix)
                pred_test_y[pred_test_y > 0.999] = 0.999
                pred_test_y[pred_test_y < 0] = 0
                pred_test_dut_wp = get_wp_list(test_df['duration'].tolist(), pred_test_y)

                test_df['dut_wp'] = np.asarray(pred_test_dut_wp)

                # print('>>> Ridge MAE on test set: {0:.4f}'.format(mean_absolute_error(test_df.true_wp, test_df.ut_wp)))
                # print('>>> Ridge R2 on test set: {0:.4f}'.format(r2_score(test_df.true_wp, test_df.ut_wp)))
                # print('=' * 79)
                # print()
            # if not, set as 'NA'
            else:
                pred_test_dut_wp = ['NA'] * test_df.shape[0]
                test_df['dut_wp'] = np.asarray(pred_test_dut_wp)

            predict_result_dict.update(test_df.set_index('vid')['dut_wp'].to_dict())

    # write to txt file
    to_write = True
    if to_write:
        output_path = 'norm_predict_results/predict_dut.p'
        print('>>> Number of videos in final test result dict: {0}'.format(len(predict_result_dict)))
        pickle.dump(predict_result_dict, open(output_path, 'wb'))
