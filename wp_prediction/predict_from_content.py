#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os
import pandas as pd
import numpy as np
import glob
import cPickle as pickle
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor


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

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    predict_result_dict = {}

    train_data = glob.glob(os.path.join('../../data/production_data/random_norm/train_data', '*.txt'))
    test_data = glob.glob(os.path.join('../../data/production_data/random_norm/test_data', '*.txt'))

    train_df = pd.concat((pd.read_csv(f, sep='\t', header=0,
                                      names=['vid', 'duration', 'definition', 'category', 'lang', 'channel', 'topics',
                                             'total_view', 'true_wp', 'wp_percentile'],
                                      dtype={'duration': int, 'definition': int, 'category': int,
                                             'wp_percentile': float}) for f in train_data))

    test_df = pd.concat((pd.read_csv(f, sep='\t', header=0,
                                     names=['vid', 'duration', 'definition', 'category', 'lang', 'channel', 'topics',
                                            'total_view', 'true_wp', 'wp_percentile'],
                                     dtype={'duration': int, 'definition': int, 'category': int,
                                            'wp_percentile': float}) for f in test_data))

    print('>>> Finish loading all data!')
    train_num = train_df.shape[0]
    print('>>> Number of training observations: {0}'.format(train_num))
    data_df = train_df.append(test_df)

    # data_df['duration'] = np.log10(data_df['duration'])

    enc_label = LabelEncoder()
    data_df['lang'] = enc_label.fit_transform(data_df['lang'])

    train_df = data_df[:train_num]
    test_df = data_df[train_num:]

    cols = ['duration', 'definition', 'category', 'lang']
    rf_regressor = RandomForestRegressor(n_estimators=10, min_samples_leaf=100, random_state=42)
    rf_regressor.fit(train_df[cols], train_df.wp_percentile)

    print('>>> Feature importances of duration, definition, category, language: {0}'.format(rf_regressor.feature_importances_))
    print('>>> Number of features: {0}'.format(rf_regressor.n_features_))

    pred_train_y = rf_regressor.predict(train_df[cols])
    pred_train_dc_wp = get_wp_list(train_df['duration'].tolist(), pred_train_y)

    pred_test_y = rf_regressor.predict(test_df[cols])
    pred_test_dc_wp = get_wp_list(test_df['duration'].tolist(), pred_test_y)

    test_df['content_wp'] = np.asarray(pred_test_dc_wp)
    pred_result = test_df[['vid', 'content_wp']]

    predict_result_dict.update(test_df.set_index('vid')['content_wp'].to_dict())

    print('>>> Random forest MAE on train set: {0:.4f}'.format(mean_absolute_error(train_df.true_wp, pred_train_dc_wp)))
    print('>>> Random forest R2 on train set: {0:.4f}'.format(r2_score(train_df.true_wp, pred_train_dc_wp)))
    print('>>> Random forest MAE on test set: {0:.4f}'.format(mean_absolute_error(test_df.true_wp, pred_test_dc_wp)))
    print('>>> Random forest R2 on test set: {0:.4f}'.format(r2_score(test_df.true_wp, pred_test_dc_wp)))
    print('='*79)
    print()

    # write to txt file
    to_write = True
    if to_write:
        output_path = 'norm_predict_results/predict_dc.p'
        print('>>> Number of videos in final test result dict: {0}'.format(len(predict_result_dict)))
        pickle.dump(predict_result_dict, open(output_path, 'wb'))
