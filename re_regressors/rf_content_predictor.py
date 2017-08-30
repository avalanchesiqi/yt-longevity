#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Predict relative engagement from content."""

from __future__ import division, print_function
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import pandas as pd
import numpy as np
import glob
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

from utils.helper import write_dict_to_pickle


def random_forest(train_x, train_y, test_x):
    # random forest regression
    rf_regressor = RandomForestRegressor(n_estimators=10, min_samples_leaf=100)
    rf_regressor.fit(train_x, train_y)
    test_yhat = rf_regressor.predict(test_x)
    return test_yhat


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    predict_result_dict = {}

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    train_data = glob.glob(os.path.join('../../production_data/tweeted_dataset_norm/train_data', '*.txt'))
    test_data = glob.glob(os.path.join('../../production_data/tweeted_dataset_norm/test_data', '*.txt'))

    train_df = pd.concat((pd.read_csv(f, sep='\t', header=0,
                                      names=['vid', 'publish', 'duration', 'definition', 'category', 'lang', 'channel',
                                             'topics', 'topics_num', 'view30', 'watch30', 'wp30', 're30', 'view120',
                                             'watch120', 'wp120', 're120', 'days', 'daily_view', 'daily_watch'],
                                      usecols=['vid', 'duration', 'definition', 'category', 'lang', 're30'],
                                      dtype={'duration': int, 'definition': int, 'category': int, 're30': float})
                          for f in train_data))

    test_df = pd.concat((pd.read_csv(f, sep='\t', header=0,
                                     names=['vid', 'publish', 'duration', 'definition', 'category', 'lang', 'channel',
                                            'topics', 'topics_num', 'view30', 'watch30', 'wp30', 're30', 'view120',
                                            'watch120', 'wp120', 're120', 'days', 'daily_view', 'daily_watch'],
                                     usecols=['vid', 'duration', 'definition', 'category', 'lang', 're30'],
                                     dtype={'duration': int, 'definition': int, 'category': int, 're30': float})
                        for f in test_data))

    train_num = train_df.shape[0]
    test_num = test_df.shape[0]
    data_df = train_df.append(test_df)

    print('>>> Finish loading all data!')
    print('>>> Number of train observations: {0}'.format(train_num))
    print('>>> Number of test observations: {0}'.format(test_num))

    data_df['log_duration'] = data_df['duration'].apply(np.log10)

    lang_le = LabelEncoder()
    data_df['enc_lang'] = lang_le.fit_transform(data_df['lang'].values)

    print(data_df.head())
    train_df = data_df[:train_num]
    test_df = data_df[train_num:]

    cols = ['log_duration', 'definition', 'category', 'enc_lang']
    target = 're30'

    # log duration, definition, category, encoded lang on random forest
    predict_re = random_forest(train_df[cols], train_df[target], test_df[cols])
    print('>>> Predict relative engagement on content with random forest regressor...')
    print('>>> MAE on test set: {0:.4f}'.format(mean_absolute_error(test_df[target].values, predict_re)))
    print('>>> R2 on test set: {0:.4f}'.format(r2_score(test_df[target].values, predict_re)))
    print('=' * 79)

    test_df = test_df.assign(pred_re=pd.Series(predict_re).values)
    predict_result_dict = test_df.set_index('vid')['pred_re'].to_dict()

    # write to pickle file
    to_write = True
    if to_write:
        print('>>> Prepare to write to pickle file...')
        print('>>> Number of videos in final test result dict: {0}'.format(len(predict_result_dict)))
        write_dict_to_pickle(dict=predict_result_dict, path='./output/content_predictor.p')
