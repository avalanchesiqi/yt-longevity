#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Predict relative engagement from topic features, with ridge regression and sparse matrix."""

from __future__ import division, print_function
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from utils.helper import write_dict_to_pickle


def _load_data(filepath):
    matrix = []
    with open(filepath, 'r') as fin:
        fin.readline()
        for line in fin:
            vid, _, duration, _, _, _, _, topics, _, _, _, _, re30, _ = line.rstrip().split('\t', 13)
            row = [vid, duration, topics, re30]
            matrix.append(row)
    print('>>> Finish loading file {0}!'.format(filepath))
    return matrix


def _build_sparse_matrix(row_idx, duration, topics, topic_dict):
    row_list = []
    col_list = []
    value_list = []

    row_list.append(row_idx)
    col_list.append(0)
    value_list.append(np.log10(int(duration)))

    topics = topics.split(',')
    for topic in topics:
        if topic in topic_dict:
            row_list.append(row_idx)
            col_list.append(1 + topic_dict[topic])
            value_list.append(1)
        else:
            return [], [], []
    return row_list, col_list, value_list


def vectorize_train_data(data):
    train_row = []
    train_col = []
    train_value = []
    train_y = []
    topic_dict = {}
    topic_cnt = 0
    row_idx = 0

    for _, _, topics, _ in data:
        if not topics == '' and topics != 'NA':
            topics = topics.split(',')
            for topic in topics:
                if topic not in topic_dict:
                    topic_dict[topic] = topic_cnt
                    topic_cnt += 1

    for _, duration, topics, re30 in data:
        if not topics == '' and topics != 'NA':
            row_list, col_list, value_list = _build_sparse_matrix(row_idx, duration, topics, topic_dict)
            train_row.extend(row_list)
            train_col.extend(col_list)
            train_value.extend(value_list)
            train_y.append(float(re30))
            row_idx += 1
    return coo_matrix((train_value, (train_row, train_col)), shape=(row_idx, topic_cnt+1)), train_y, topic_dict


def vectorize_test_data(data, topic_dict):
    test_vids = []
    test_row = []
    test_col = []
    test_value = []
    test_y = []
    n_topic = len(topic_dict)
    row_idx = 0

    for vid, duration, topics, re30 in data:
        if not topics == '' and topics != 'NA':
            row_list, col_list, value_list = _build_sparse_matrix(row_idx, duration, topics, topic_dict)
            test_row.extend(row_list)
            test_col.extend(col_list)
            test_value.extend(value_list)
            test_y.append(float(re30))
            row_idx += 1
            test_vids.append(vid)
    return coo_matrix((test_value, (test_row, test_col)), shape=(row_idx, n_topic + 1)), test_y, test_vids


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    # estimators = [("Ridge", Ridge())]

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    train_loc = '../../production_data/tweeted_dataset_norm/train_data'
    test_loc = '../../production_data/tweeted_dataset_norm/test_data'

    train_cv_matrix = []
    print('>>> Start to load training dataset...')
    for subdir, _, files in os.walk(train_loc):
        for f in files:
            train_cv_matrix.extend(_load_data(os.path.join(subdir, f)))
    train_cv_matrix = np.array(train_cv_matrix)

    # split into 80/20 train, cv matrix
    train_matrix, cv_matrix = train_test_split(train_cv_matrix, train_size=0.8, test_size=0.2, random_state=42)
    n_train = len(train_matrix)
    n_cv = len(cv_matrix)

    test_matrix = []
    print('>>> Start to load test dataset...')
    for subdir, _, files in os.walk(test_loc):
        for f in files:
            test_matrix.extend(_load_data(os.path.join(subdir, f)))
    test_matrix = np.array(test_matrix)
    n_test = len(test_matrix)

    print('\n', '-'*79, '\n')

    # generate train dataset on the fly
    train_sparse_x, train_y, train_topics = vectorize_train_data(train_matrix)
    print('>>> Loading training sparse matrix, number of videos: {0}'.format(len(train_y)))
    cv_sparse_x, cv_y, _ = vectorize_test_data(cv_matrix, train_topics)
    print('>>> Loading cv sparse matrix, number of videos: {0}'.format(len(cv_y)))

    # grid search over alpha in ridge regressor
    search_alpha_array = [10**t for t in range(-5, 5)]
    cv_mae = []
    for search_alpha in search_alpha_array:
        estimator = Ridge(alpha=search_alpha)
        estimator.fit(train_sparse_x, train_y)
        print('>>> Finish fitting estimator...')
        n_topic = len(train_topics)
        print('>>> Start to predict cv sparse matrix...')
        cv_yhat = estimator.predict(cv_sparse_x)
        cv_mae.append(mean_absolute_error(cv_y, cv_yhat))
        print('>>> CV phase, MAE: {0} with alpha value: {1}'.format(cv_mae, search_alpha))
        print('='*79, '\n')

    # build the best estimator
    best_alpha_idx = np.argmin(np.array(cv_mae))
    best_alpha = search_alpha_array[best_alpha_idx]
    print('>>> best hyper parameter alpha idx: {0}'.format(best_alpha_idx))
    print('>>> best hyper parameter alpha: {0}'.format(best_alpha))
    best_estimator = Ridge(alpha=best_alpha)
    train_cv_sparse_x, train_cv_y, train_cv_topics = vectorize_train_data(train_cv_matrix)
    best_estimator.fit(train_cv_sparse_x, train_cv_y)

    # build test dataset on the fly
    test_sparse_x, test_y, test_vids = vectorize_test_data(test_matrix, train_cv_topics)
    test_yhat = best_estimator.predict(test_sparse_x)
    print('>>> predict {0} videos in test dataset'.format(len(test_yhat)))
    print('>>> Ridge sparse model: MAE of test dataset: {0}'.format(mean_absolute_error(test_y, test_yhat)))

    predict_result_dict = {}
    for j in xrange(len(test_vids)):
        predict_result_dict[test_vids[j]] = test_yhat[j]

    # write to pickle file
    to_write = True
    if to_write:
        print('>>> Prepare to write to pickle file...')
        print('>>> Number of videos in final test result dict: {0}'.format(len(predict_result_dict)))
        write_dict_to_pickle(dict=predict_result_dict, path='./output/sparse_topic_predictor.p')
