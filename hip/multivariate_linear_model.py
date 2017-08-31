#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Forecast future temporal attention, viewership or watch time. Multivariate Linear model from Pinto WSDM'13"""

from __future__ import print_function, division
import sys, os, bz2, json
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import cPickle as pickle
from collections import defaultdict
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge

from utils.helper import strify


if __name__ == '__main__':
    # == == == == == == == == Part 1: Load ACTIVE dataset == == == == == == == == #
    # First time it gets loaded from the JSON format and writes essential fields into a pickle binary file.
    # check if the binary exists
    if not os.path.exists('./active-dataset.p'):
        print('>>> Converting ACTIVE dataset from JSON format to pickle... might take a while!')
        test_cases = {}
        with bz2.BZ2File('./active-dataset.json.bz2') as f:
            dataset = json.loads(f.readline())
            for video in dataset:
                if not video['duration'] == 'NA':
                    test_cases[video['YoutubeID']] = (video['numShare'], video['dailyViewcount'], video['watchTime'], video['duration'])
        pickle.dump(test_cases, open('./active-dataset.p', 'wb'))

    print('>>> Loading the ACTIVE dataset from pickle...')
    test_cases = pickle.load(open('./active-dataset.p', 'rb'))
    test_vids = test_cases.keys()

    # == == == == == == == == Part 2: Set up experiment parameters == == == == == == == == #
    age = 120
    num_train = 90
    predict_results = defaultdict(list)
    with_share = 0
    use_view = 1
    if not os.path.exists('./output'):
        os.mkdir('./output')
    output_path = './output/mlr_forecast_{0}_{1}_share.txt'.format(['watch', 'view'][use_view], ['without', 'with'][with_share])
    print('>>> Forecast daily {0} {1} share series\n'.format(['watch', 'view'][use_view], ['without', 'with'][with_share]))

    # == == == == == == == == Part 3: Prepare numpy data matrix == == == == == == == == #
    attention_data = []
    share_data = []
    vid_array = []
    for vid in test_vids:
        dailyshare, dailyview, dailywatch, duration = test_cases[vid]
        # first 120 days, select view count or watch time as dependent variable
        daily_attention = [dailywatch, dailyview][use_view][:age]
        daily_share = dailyshare[:age]
        if len(daily_attention) == 120 and len(daily_share) == 120:
            attention_data.append(daily_attention)
            share_data.append(daily_share)
            vid_array.append(vid)

    # convert to ndarray
    attention_data = np.array(attention_data)
    share_data = np.array(share_data)
    vid_array = np.array(vid_array)

    # == == == == == == == == Part 4: Forecast future attention == == == == == == == == #
    # 10-repeated 10-fold cross validation
    rkf = RepeatedKFold(n_splits=5, n_repeats=5)

    fold_idx = 0
    for train_idx, test_idx in rkf.split(vid_array):
        fold_idx += 1
        print('>>> Forecast on fold: {0}'.format(fold_idx))
        x_train_predict = attention_data[train_idx, :num_train]
        x_test_predict = attention_data[test_idx, :num_train]
        m, n = len(train_idx), len(test_idx)
        # iterate over forecast days
        for i in xrange(num_train, age):
            print('predict for day', i)
            if with_share == 1:
                x_train = np.hstack((x_train_predict, share_data[train_idx, :i+1]))
                x_test = np.hstack((x_test_predict, share_data[test_idx, :i+1]))
            else:
                x_train = x_train_predict
                x_test = x_test_predict
            print('shape of x_track', x_train.shape)
            norm = np.hstack((x_train[:, :i], attention_data[train_idx, i].reshape(m, 1)))
            x_train_norm = x_train/np.sum(norm, axis=1)[:, None]
            y_train = np.ones(m, )

            # == == == == == == == == Part 5: Training with Ridge Regression == == == == == == == == #
            ridge_model = Ridge(fit_intercept=False, alpha=0.1)
            ridge_model.fit(x_train_norm, y_train)
            predict_train_value = (ridge_model.predict(x_train)-np.sum(x_train[:, :i], axis=1)).reshape(m, 1)
            predict_train_value[predict_train_value < 0] = 0
            print(predict_train_value[:5].flatten())
            print(attention_data[train_idx, i].flatten()[:5])
            x_train_predict = np.hstack((x_train_predict, predict_train_value))
            predict_test_value = (ridge_model.predict(x_test)-np.sum(x_test[:, :i], axis=1)).reshape(n, 1)
            predict_test_value[predict_test_value < 0] = 0
            x_test_predict = np.hstack((x_test_predict, predict_test_value))
            print('---------')
            print(predict_test_value[:5].flatten())
            print(attention_data[test_idx, i].flatten()[:5])

        for i, j in enumerate(test_idx):
            predict_results[vid_array[j]].append(x_test_predict[i, num_train:])

    # aggregate predict values from folds
    with open(output_path, 'w') as fout:
        for vid in vid_array:
            predicted_daily_watchtime = np.mean(np.array(predict_results[vid]), axis=0)
            fout.write('{0},'.format(vid))
            fout.write('{0}\n'.format(strify(predicted_daily_watchtime)))
