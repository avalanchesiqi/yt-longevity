from __future__ import print_function, division
import sys
import os
import bz2
import json
import cPickle as pickle
from collections import defaultdict
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression

# Multivariate Linear model from Pinto WSDM'13


def strify(iterable_struct):
    """
    Convert an iterable structure to comma separated string
    :param iterable_struct: an iterable structure
    :return: a string with comma separated
    """
    return ','.join(map(str, iterable_struct))


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
    # setting parameters
    age = 120
    num_train = 90
    predict_results = defaultdict(list)
    with_share = False
    use_view = False
    if with_share and use_view:
        # 0 is daily view, 1 is daily watch
        data_idx = 0
        output_path = '../engagement/data/mlr_daily_forecast_view_w_share.txt'
    elif with_share and not use_view:
        data_idx = 1
        output_path = '../engagement/data/mlr_daily_forecast_watch_w_share.txt'
    elif not with_share and use_view:
        data_idx = 0
        output_path = '../engagement/data/mlr_daily_forecast_view_wo_share.txt'
    else:
        data_idx = 1
        output_path = '../engagement/data/mlr_daily_forecast_watch_wo_share.txt'

    attention_data = []
    share_data = []
    vid_array = []
    for tc_idx, vid in enumerate(test_vids):
        dailyshare, dailyview, dailywatch, duration = test_cases[vid]

        # first 120 days
        daily_attention = [dailyview, dailywatch][data_idx][:age]
        daily_share = dailyshare[:age]

        # select view count or watch time as dependent variable
        if len(daily_attention) == 120 and len(daily_share) == 120:
            attention_data.append(daily_attention)
            share_data.append(daily_share)
            vid_array.append(vid)

    # convert to ndarray
    attention_data = np.array(attention_data)
    share_data = np.array(share_data)
    vid_array = np.array(vid_array)

    # 10-repeated 10-fold cross validation
    rkf = RepeatedKFold(n_splits=10, n_repeats=10)

    epoch = 0
    for train_idx, test_idx in rkf.split(attention_data):
        epoch += 1
        print('>>> epoch: {0}'.format(epoch))
        x_train_predict = attention_data[train_idx, :num_train]
        x_test_predict = attention_data[test_idx, :num_train]
        m, n = len(train_idx), len(test_idx)
        # iterate over forecast days
        for i in xrange(num_train, age):
            if with_share:
                x_train = np.hstack((x_train_predict, share_data[train_idx, :i+1]))
                x_test = np.hstack((x_test_predict, share_data[test_idx, :i+1]))
            else:
                x_train = x_train_predict
                x_test = x_test_predict
            y_train = attention_data[train_idx, i]

            # == == == == == == == == Part 3: Training with an OLS regression == == == == == == == == #
            lr_model = LinearRegression(fit_intercept=False)
            lr_model.fit(x_train, y_train)
            predict_train_value = lr_model.predict(x_train).reshape(m, 1)
            predict_train_value[predict_train_value < 0] = 0
            x_train_predict = np.hstack((x_train_predict, predict_train_value))
            predict_test_value = lr_model.predict(x_test).reshape(n, 1)
            predict_test_value[predict_test_value < 0] = 0
            x_test_predict = np.hstack((x_test_predict, predict_test_value))

        for i, j in enumerate(test_idx):
            predict_results[vid_array[j]].append(x_test_predict[i, num_train:])

    # aggregate predict values from folds
    with open(output_path, 'w') as fout:
        for vid in vid_array:
            predicted_daily_watchtime = np.mean(np.array(predict_results[vid]), axis=0)
            fout.write('{0},'.format(vid))
            fout.write('{0}\n'.format(strify(predicted_daily_watchtime)))
