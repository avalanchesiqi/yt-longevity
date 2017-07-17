from __future__ import print_function, division
import sys
import os
import bz2
import json
import cPickle as pickle
import numpy as np
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
    lr_models = []

    matrix_data = []
    vid_list = []
    for tc_idx, vid in enumerate(test_vids):
        dailyshare, dailyview, watchtime, duration = test_cases[vid]

        # first 120 days
        dailyview = dailyview[:age]
        watchtime = watchtime[:age]

        # select view count or watch time as dependent variable
        row_data = watchtime
        if len(row_data) == 120:
            matrix_data.append(row_data)
            vid_list.append(vid)

    # convert to ndarray
    matrix_data = np.array(matrix_data)
    m = matrix_data.shape[0]
    predict_result = np.array(matrix_data[:, :num_train])

    # iterate over forecast days
    for i in xrange(num_train, age):
        x_train = matrix_data[:, :i]/(np.sum(matrix_data[:, :i+1], axis=1).reshape(m, 1))
        y_train = np.ones((m, 1))

        # == == == == == == == == Part 3: Training with an OLS regression == == == == == == == == #
        lr_model = LinearRegression(fit_intercept=False)
        print('>>> Training with OLS regression for day {0}...'.format(i+1))
        lr_model.fit(x_train, y_train)
        lr_models.append(lr_model)

    for i in xrange(num_train, age):
        predict_value = lr_models[i-num_train].predict(predict_result[:, :i]).reshape(m, 1)
        forecast_next_day = predict_value - np.sum(predict_result[:, :i], axis=1).reshape(m, 1)
        forecast_next_day[forecast_next_day < 0] = 0
        predict_result = np.hstack((predict_result, forecast_next_day))

    with open('mlr_daily_forecast_watch.log', 'w') as fout:
        for i in xrange(m):
            fout.write('{0},'.format(vid_list[i]))
            fout.write('{0},'.format(np.sum(predict_result[i, :num_train])))
            fout.write('{0}\n'.format(strify(predict_result[i, num_train: age])))

    # np.savetxt('mlr_view2.log', true_predict_matrix, delimiter='\t', newline='\n')
