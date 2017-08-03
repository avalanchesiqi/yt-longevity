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


def random_forest(cols, target):
    # random forest regression
    rf_regressor = RandomForestRegressor(n_estimators=10, min_samples_leaf=100)
    rf_regressor.fit(train_df[cols], train_df[target])

    print('>>> Feature importances of duration, definition, category, language: {0}'.format(
        rf_regressor.feature_importances_))
    print('>>> Number of features: {0}'.format(rf_regressor.n_features_))

    test_yhat = rf_regressor.predict(test_df[cols])
    if target == 'ep30':
        test_dc_wp = get_wp(test_df['duration'].tolist(), test_yhat)
    else:
        test_dc_wp = test_yhat

    print('>>> MAE on test set: {0:.4f}'.format(mean_absolute_error(test_df['wp30'], test_dc_wp)))
    print('>>> R2 on test set: {0:.4f}'.format(r2_score(test_df['wp30'], test_dc_wp)))
    print('=' * 79)
    print()

    return test_dc_wp


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    dur_engage_str_map = pickle.load(open('../engagement/data/tweeted_dur_engage_map.p', 'rb'))
    dur_engage_map = {key: list(map(float, value.split(','))) for key, value in dur_engage_str_map.items()}

    duration_split_points = np.array(dur_engage_map['duration'])

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    predict_result_dict = {}

    train_data = glob.glob(os.path.join('../../production_data/tweeted_dataset_norm/train_data', '*.txt'))
    test_data = glob.glob(os.path.join('../../production_data/tweeted_dataset_norm/test_data', '*.txt'))

    train_df = pd.concat((pd.read_csv(f, sep='\t', header=0,
                                      names=['vid', 'publish', 'duration', 'definition', 'category', 'lang', 'channel',
                                             'topics', 'topics_num', 'view30', 'watch30', 'wp30', 'ep30', 'view120',
                                             'watch120', 'wp120', 'ep120', 'days', 'daily_view', 'daily_watch'],
                                      dtype={'duration': int, 'definition': int, 'category': int, 'wp30': float,
                                             'ep30': float}) for f in train_data))

    test_df = pd.concat((pd.read_csv(f, sep='\t', header=0,
                                     names=['vid', 'publish', 'duration', 'definition', 'category', 'lang', 'channel',
                                            'topics', 'topics_num', 'view30', 'watch30', 'wp30', 'ep30', 'view120',
                                            'watch120', 'wp120', 'ep120', 'days', 'daily_view', 'daily_watch'],
                                     dtype={'duration': int, 'definition': int, 'category': int, 'wp30': float,
                                            'ep30': float}) for f in test_data))

    train_num = train_df.shape[0]
    test_num = test_df.shape[0]
    print('>>> Finish loading all data!')
    print('>>> Number of train observations: {0}'.format(train_num))
    print('>>> Number of test observations: {0}'.format(test_num))
    print()
    data_df = train_df.append(test_df)

    data_df['log_duration'] = np.log10(data_df['duration'])

    enc_label = LabelEncoder()
    data_df['lang'] = enc_label.fit_transform(data_df['lang'])

    train_df = data_df[:train_num]
    test_df = data_df[train_num:]

    lin_cols = ['duration', 'definition', 'category', 'lang']
    log_cols = ['log_duration', 'definition', 'category', 'lang']
    percentile_target = 'ep30'
    percentage_target = 'wp30'

    # random forest, duration, definition, category, lang on percentile
    print('>>> Random forest, duration linear ~ percentile')
    random_forest(lin_cols, percentile_target)

    # random forest, log duration, definition, category, lang on percentile
    print('>>> Random forest, duration log ~ percentile')
    output_predict = random_forest(log_cols, percentile_target)

    # random forest, duration, definition, category, lang on percentage
    print('>>> Random forest, duration linear ~ percentage')
    random_forest(lin_cols, percentage_target)

    # random forest, duration log, definition, category, lang on percentage
    print('>>> Random forest, duration log ~ percentage')
    random_forest(log_cols, percentage_target)

    test_df['content_wp'] = np.asarray(output_predict)
    pred_result = test_df[['vid', 'content_wp']]
    predict_result_dict.update(test_df.set_index('vid')['content_wp'].to_dict())

    # write to p file
    to_write = True
    if to_write:
        output_path = '../engagement/data/wp_prediction/predict_dc.p'
        print('>>> Number of videos in final test result dict: {0}'.format(len(predict_result_dict)))
        pickle.dump(predict_result_dict, open(output_path, 'wb'))
