#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os
import numpy as np
import cPickle as pickle
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge


def get_wp(duration, percentile):
    # map wp percentile to watch percentage
    if len(duration) > 1:
        wp_list = []
        for d, p in zip(duration, percentile):
            wp_list.extend(get_wp([d], [p]))
        return wp_list
    else:
        bin_idx = np.sum(duration_split_points < duration[0])
        duration_bin = dur_engage_map[bin_idx]
        percentile = int(round(percentile[0] * 1000))
        wp_percentile = duration_bin[percentile]
        return [wp_percentile]


def ridge_regression(train_x, train_y, test_x, target):
    # ridge regression
    ridge_model = Ridge(fit_intercept=True)
    ridge_model.fit(train_x, train_y)
    test_yhat = ridge_model.predict(test_x).flatten()

    if target == 'percentile':
        test_yhat[test_yhat > 0.999] = 0.999
        test_yhat[test_yhat < 0] = 0
        test_du_wp = get_wp(test_duration, test_yhat)
    else:
        test_du_wp = test_yhat

    return list(test_du_wp)


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

    # train_loc = '../../data/production_data/random_channel/train_data'
    train_loc = '../../production_data/random_channel/train_data'
    # test_loc = '../../data/production_data/random_channel/test_data'
    test_loc = '../../production_data/random_channel/test_data'

    true_value = []
    lin_percentile_value = []
    log_percentile_value = []
    lin_wp_value = []
    log_wp_value = []

    for subdir, _, files in os.walk(test_loc):
        for f in files:
            # if we have observed this channel before, minimal observations: 5
            if f in train_channel_cnt_map and train_channel_cnt_map[f] > 4:
                sub_f = f[:4]
                train_data_path = os.path.join(train_loc, sub_f, f)
                train_lines = open(train_data_path, 'r').readlines()

                # get category encoding
                category_dict = {}
                category_cnt = 0
                for category in [x.split('\t')[3] for x in train_lines]:
                    if category not in category_dict:
                        category_dict[category] = category_cnt
                        category_cnt += 1

                # get lang encoding
                lang_dict = {}
                lang_cnt = 0
                for lang in [x.split('\t')[4] for x in train_lines]:
                    if lang not in lang_dict:
                        lang_dict[lang] = lang_cnt
                        lang_cnt += 1

                # get topic encoding
                topic_dict = {}
                topic_cnt = 0
                for video_topics in [x.split('\t')[6] for x in train_lines]:
                    for topic in video_topics.split(','):
                        if topic not in topic_dict:
                            topic_dict[topic] = topic_cnt
                            topic_cnt += 1

                # get channel history
                train_matrix = []
                with open(train_data_path, 'r') as fin:
                    for line in fin:
                        row = np.zeros(2+category_cnt+lang_cnt+topic_cnt+3)
                        _, duration, definition, category, lang, _, topics, _, true_wp, wp_percentile = line.rstrip().split('\t')
                        topics = topics.split(',')
                        if definition == '0':
                            row[0] = 1
                        else:
                            row[1] = 1
                        row[2+category_dict[category]] = 1
                        row[2+category_cnt+lang_dict[lang]] = 1
                        for topic in topics:
                            row[2+category_cnt+lang_cnt+topic_dict[topic]] = 1
                        row[-3] = int(duration)
                        row[-2] = float(true_wp)
                        row[-1] = float(wp_percentile)
                        train_matrix.append(row)
                train_matrix = np.array(train_matrix)

                train_lin = train_matrix[:, :-2]
                train_log = np.hstack((train_matrix[:, :-3], np.log10(train_matrix[:, -3]).reshape(-1, 1)))
                train_wp = train_matrix[:, -2].reshape(-1, 1)
                train_percentile = train_matrix[:, -1].reshape(-1, 1)

                test_matrix = []
                test_vids = []
                with open(os.path.join(subdir, f), 'r') as fin:
                    for line in fin:
                        row = np.zeros(2+category_cnt+lang_cnt+topic_cnt+2)
                        vid, duration, definition, category, lang, _, topics, _, true_wp, _ = line.rstrip().split('\t')
                        topics = topics.split(',')
                        if definition == '0':
                            row[0] = 1
                        else:
                            row[1] = 1
                        if category in category_dict:
                            row[2+category_dict[category]] = 1
                        if lang in lang_dict:
                            row[2+category_cnt+lang_dict[lang]] = 1
                        for topic in topics:
                            if topic in topic_dict:
                                row[2+category_cnt+lang_cnt+topic_dict[topic]] = 1
                        row[-2] = int(duration)
                        row[-1] = float(true_wp)
                        test_matrix.append(row)
                        test_vids.append(vid)
                test_matrix = np.array(test_matrix)

                test_lin = test_matrix[:, :-1]
                test_log = np.hstack((test_matrix[:, :-2], np.log10(test_matrix[:, -2]).reshape(-1, 1)))
                test_wp = test_matrix[:, -1].reshape(1, -1)
                test_duration = test_matrix[:, -2]

                # ridge regression, linear duration on wp
                lin_wp_value.extend(ridge_regression(train_lin, train_wp, test_lin, 'wp'))

                # ridge regression, log duration on wp
                log_wp_value.extend(ridge_regression(train_log, train_wp, test_log, 'wp'))

                # ridge regression, linear duration on percentile
                lin_percentile_value.extend(ridge_regression(train_lin, train_percentile, test_lin, 'percentile'))

                # ridge regression, log duration on percentile
                test_duct_wp = ridge_regression(train_log, train_percentile, test_log, 'percentile')
                log_percentile_value.extend(test_duct_wp)

                true_value.extend(test_wp.tolist()[0])
                current_trained_num = len(true_value)
                if current_trained_num % 1000 == 0:
                    print('>>> Current finished videos: {0}...'.format(current_trained_num))

            # if not, set as 'NA'
            else:
                test_vid = []
                test_lines = open(os.path.join(subdir, f), 'r').readlines()
                test_num = len(test_lines)
                test_vids = [x.split('\t', 1)[0] for x in test_lines]
                test_duct_wp = ['NA'] * test_num

            for vid, pred_wp in zip(test_vids, test_duct_wp):
                predict_result_dict[vid] = pred_wp

    print('>>> linear wp:')
    print('>>> MAE on test set: {0:.4f}'.format(mean_absolute_error(true_value, lin_wp_value)))
    print('>>> R2 on test set: {0:.4f}'.format(r2_score(true_value, lin_wp_value)))
    print('=' * 79)
    print()

    print('>>> log wp:')
    print('>>> MAE on test set: {0:.4f}'.format(mean_absolute_error(true_value, log_wp_value)))
    print('>>> R2 on test set: {0:.4f}'.format(r2_score(true_value, log_wp_value)))
    print('=' * 79)
    print()

    print('>>> linear percentile:')
    print('>>> MAE on test set: {0:.4f}'.format(mean_absolute_error(true_value, lin_percentile_value)))
    print('>>> R2 on test set: {0:.4f}'.format(r2_score(true_value, lin_percentile_value)))
    print('=' * 79)
    print()

    print('>>> log percentile:')
    print('>>> MAE on test set: {0:.4f}'.format(mean_absolute_error(true_value, log_percentile_value)))
    print('>>> R2 on test set: {0:.4f}'.format(r2_score(true_value, log_percentile_value)))
    print('=' * 79)
    print()

    # write to pickle file
    to_write = True
    if to_write:
        output_path = 'norm_predict_results/predict_duct.p'
        print('>>> Number of videos in final test result dict: {0}'.format(len(predict_result_dict)))
        pickle.dump(predict_result_dict, open(output_path, 'wb'))
