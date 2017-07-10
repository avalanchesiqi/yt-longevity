from __future__ import print_function, division
import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import model_selection, neighbors, metrics, linear_model
from scipy import stats
from scipy.interpolate import interp1d


def read_as_float_array(content, truncated=None, delimiter=None):
    """
    Read input as a float array.
    :param content: string input
    :param truncated: head number of elements extracted
    :param delimiter: delimiter string
    :return: a numpy float array
    """
    if truncated is None:
        return np.array(map(float, content.split(delimiter)), dtype=np.float64)
    else:
        return np.array(map(float, content.split(delimiter)[:truncated]), dtype=np.float64)


def predict_watch_prec(input_x):
    output_y = []
    for x in input_x:
        idx = np.sum(duration_gap <= x)
        output_y.append(mean_watch_prec[idx])
    return np.array(output_y)


def remove_invalid_value(arr):
    for i, v in enumerate(arr):
        if v > 1:
            arr[i] = 1
        elif v < 0:
            arr[i] = 0
    return arr


def knn_regression(data, idx, channel_id, to_write):
    train_size = 0.6
    cv_size = 0.8
    train_data = data[:int(train_size*len(data))]
    cv_data = data[int(train_size*len(data)): int(cv_size*len(data))]
    test_data = data[int(cv_size*len(data)):]

    train_x = train_data['video duration'].reshape(-1, 1)
    # train_x2 = train_data[['video duration', 'topic similarity']]
    cv_x = cv_data['video duration'].reshape(-1, 1)
    # cv_x2 = cv_data[['video duration', 'topic similarity']]
    test_x = test_data['video duration'].reshape(-1, 1)
    # test_x2 = test_data[['video duration', 'topic similarity']]
    train_y = train_data['watch percentage'].reshape(-1, 1)
    cv_y = cv_data['watch percentage'].reshape(-1, 1)
    test_y = test_data['watch percentage'].reshape(-1, 1)

    T = np.linspace(1, np.max(train_x, 0), 250)[:, np.newaxis]

    # mle model
    print('>>> train: {0} videos; cv: {1} videos; test: {2} videos'.format(len(train_x), len(cv_x), len(test_x)))
    mle_test_yhat = remove_invalid_value(predict_watch_prec(test_x))
    mae_mle = metrics.mean_absolute_error(test_y, mle_test_yhat)
    print('>>> MAE mle: {0:.4f}'.format(mae_mle))

    # KNN model
    weights = 'distance'
    best_knn_err = np.inf
    best_neighbors_num = None
    for n_neighbors in xrange(5, len(cv_x)):
        knn_initializer = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
        knn_model = knn_initializer.fit(train_x, train_y)
        mae_knn = metrics.mean_absolute_error(cv_y, knn_model.predict(cv_x))
        if mae_knn < best_knn_err:
            best_knn_err = mae_knn
            best_neighbors_num = n_neighbors
    best_knn_initializer = neighbors.KNeighborsRegressor(best_neighbors_num, weights=weights)
    best_knn_model = best_knn_initializer.fit(np.vstack((train_x, cv_x)), np.vstack((train_y, cv_y)))
    knn_test_yhat = remove_invalid_value(best_knn_model.predict(test_x))
    best_mae_knn = metrics.mean_absolute_error(test_y, knn_test_yhat)
    knn_y_ = best_knn_model.predict(T)
    print('>>> MAE knn: {0:.4f} @{1} neighbors'.format(best_mae_knn, best_neighbors_num))

    # LSH forest model

    # linear regression model: linear scale
    best_lin_reg_lin_error = np.inf
    best_lin_alpha = None
    for alpha in np.arange(-2, 2, 0.2):
        lin_reg_lin = linear_model.Ridge(alpha=10**alpha).fit(train_x, train_y)
        mae_ridge_lin = metrics.mean_absolute_error(cv_y, lin_reg_lin.predict(cv_x))
        if mae_ridge_lin < best_lin_reg_lin_error:
            best_lin_reg_lin_error = mae_ridge_lin
            best_lin_alpha = alpha
    best_lin_reg_lin = linear_model.Ridge(alpha=10**best_lin_alpha).fit(np.vstack((train_x, cv_x)), np.vstack((train_y, cv_y)))
    ridge_lin_test_yhat = remove_invalid_value(best_lin_reg_lin.predict(test_x))
    best_mae_ridge_lin = metrics.mean_absolute_error(test_y, ridge_lin_test_yhat)
    ridge_lin_y_ = best_lin_reg_lin.predict(T)
    print('>>> MAE ridge linear scale: {0:.4f}'.format(best_mae_ridge_lin))

    # linear regression model: log scale
    best_lin_reg_log_error = np.inf
    best_log_alpha = None
    for alpha in np.arange(-2, 2, 0.2):
        lin_reg_log = linear_model.Ridge(alpha=10**alpha).fit(np.log10(train_x), train_y)
        mae_ridge_log = metrics.mean_absolute_error(cv_y, lin_reg_log.predict(np.log10(cv_x)))
        if mae_ridge_log < best_lin_reg_log_error:
            best_lin_reg_log_error = mae_ridge_log
            best_log_alpha = alpha
    best_lin_reg_log = linear_model.Ridge(alpha=10**best_log_alpha).fit(np.vstack((np.log10(train_x), np.log10(cv_x))), np.vstack((train_y, cv_y)))
    ridge_log_test_yhat = remove_invalid_value(best_lin_reg_log.predict(np.log10(test_x)))
    best_mae_ridge_log = metrics.mean_absolute_error(test_y, ridge_log_test_yhat)
    ridge_log_y_ = best_lin_reg_log.predict(np.log10(T))
    print('>>> MAE ridge log scale: {0:.4f}'.format(best_mae_ridge_log))

    # # linear regression model
    # linreg2 = linear_model.Ridge().fit(train_x2, train_y)
    # mae_ridge2 = metrics.mean_absolute_error(test_y, linreg2.predict(test_x2))
    # print('>>> mean absolute error ridge2: {0:.4f}'.format(mae_ridge2))
    #
    # linreg3 = linear_model.Ridge(normalize=True).fit(train_x2, train_y)
    # mae_ridge3 = metrics.mean_absolute_error(test_y, linreg3.predict(test_x2))
    # print('>>> mean absolute error ridge3: {0:.4f}'.format(mae_ridge3))

    # # lowess
    # frac_num = 50/len(train_x)
    # lowess = sm.nonparametric.lowess(train_y.ravel(), train_x.ravel(), frac=frac_num)
    # # unpack the lowess smoothed points to their values
    # lowess_x = list(zip(*lowess))[0]
    # lowess_y = list(zip(*lowess))[1]
    # # run scipy's interpolation. There is also extrapolation I believe
    # f = interp1d(lowess_x, lowess_y, bounds_error=False)
    # y___ = f(T)
    # mae_lowess = metrics.mean_absolute_error(test_y, f(test_x))
    mae_lowess = 1
    # print('>>> mean absolute error lowess: {0:.4f}'.format(mae_lowess))

    # ax1 = fig.add_subplot(221 + idx)
    # ax1.plot(duration_gap, mean_watch_prec, 'mo-', ms=1, label='global: {0:.4f}'.format(mae_mle))
    # ax1.scatter(train_x, train_y, c='k', s=10)
    # ax1.plot(T, knn_y_, c='g', label='knn@{0}: {1:.4f}'.format(best_neighbors_num, best_mae_knn))
    # ax1.plot(T, ridge_lin_y_, '--', c='b', label='lin reg lin: {0:.4f}'.format(best_mae_ridge_lin))
    # ax1.plot(T, ridge_log_y_, '-', c='b', label='lin reg log: {0:.4f}'.format(best_mae_ridge_log))
    # # ax1.plot(T, y___, '.--', c='y', ms=3, label='lowess')
    # ax1.scatter(test_x, test_y, c='r', s=10, marker='x')
    # ax1.set_xlabel('Video Duration')
    # ax1.set_ylabel('Watch Percentage')
    # ax1.set_xlim([0, max(max(train_x), max(test_x))])
    # ax1.set_ylim([0, 1])
    # ax1.set_title(channel_id)
    # # ax1.text(0, 0.75, 'global: {0:.4f}\nknn@{1}: {2:.4f}\nridge lin@10^{3}: {4:.4f}\nridge log@10^{5}: {6:.4f}\nlowess: {7:.4f}'
    # #          .format(mae_mle, best_neighbors_num, best_mae_knn, best_lin_alpha, best_mae_ridge_lin, best_log_alpha, best_mae_ridge_log, mae_lowess))
    # plt.legend(loc='best')
    # print('-'*79)

    predict_matrix = [test_y.ravel(), mle_test_yhat.ravel(), knn_test_yhat.ravel(),
                      ridge_lin_test_yhat.ravel(), ridge_log_test_yhat.ravel(), to_write.ravel()]
    with open('prediction.log', 'a') as logout:
        for i in xrange(len(test_y)):
            logout.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(predict_matrix[0][i], predict_matrix[1][i], predict_matrix[2][i],
                                                            predict_matrix[3][i], predict_matrix[4][i], predict_matrix[5][i]))


def train_with_topics(filepath):
    data_mat = []
    channel_topics = defaultdict(int)
    idx = 0
    topic_lookup = {}

    # get topic dict
    with open(filepath, 'r') as fin:
        for line in fin:
            _, duration, _, topics, watch_prec = line.rstrip().split('\t')
            topics = topics.split(',')
            for topic in topics:
                channel_topics[topic] += 1
                if topic not in topic_lookup:
                    topic_lookup[topic] = idx
                    idx += 1
    # get data matrix
    with open(filepath, 'r') as fin:
        for line in fin:
            _, duration, _, topics, watch_prec = line.rstrip().split('\t')
            topics = topics.split(',')
            vector_data = [np.log10(int(duration))]
            vector_data.extend([0]*idx)
            for topic in topics:
                vector_data[topic_lookup[topic] + 1] = 1
            vector_data.append(float(watch_prec))
            data_mat.append(np.array(vector_data, dtype=np.float64))
    data_mat = np.array(data_mat)
    train_size = 0.6
    cv_size = 0.8
    train_data = data_mat[:int(train_size*len(data_mat))]
    train_x = train_data[:, :-1]
    train_y = train_data[:, -1].reshape(-1, 1)
    cv_data = data_mat[int(train_size*len(data_mat)): int(cv_size*len(data_mat))]
    cv_x = cv_data[:, :-1]
    cv_y = cv_data[:, -1].reshape(-1, 1)
    test_data = data_mat[int(cv_size*len(data_mat)):]
    test_x = test_data[:, :-1]
    test_y = test_data[:, -1].reshape(-1, 1)

    best_topic_error = np.inf
    best_topic_alpha = None
    for alpha in np.arange(-2, 2, 0.2):
        topic_model = linear_model.Ridge(alpha=10**alpha).fit(train_x, train_y)
        mae_topic_cv = metrics.mean_absolute_error(cv_y, topic_model.predict(cv_x))
        if mae_topic_cv < best_topic_error:
            best_topic_error = mae_topic_cv
            best_topic_alpha = alpha
    best_topic_model = linear_model.Ridge(alpha=10**best_topic_alpha).fit(np.vstack((train_x, cv_x)), np.vstack((train_y, cv_y)))
    topic_test_yhat = remove_invalid_value(best_topic_model.predict(test_x))
    best_mae_topic = metrics.mean_absolute_error(test_y, topic_test_yhat)
    print('>>> MAE topic regression: {0:.4f}'.format(best_mae_topic))
    return topic_test_yhat


if __name__ == '__main__':
    fig = plt.figure(figsize=(14, 10))

    fin = open('global_parameters.txt', 'r')
    duration_gap = read_as_float_array(fin.readline().rstrip(), delimiter=',')
    mean_watch_prec = read_as_float_array(fin.readline().rstrip(), delimiter=',')
    fin.close()

    input_doc = '../../data/production_data/active_channels_feature'

    global_errors = []
    knn_errors = []
    linreg_errors = []
    logreg_errors = []

    fig_idx = 0
    for subdir, _, files in os.walk(input_doc):
        # for f in random.sample(files, 4):
        for f in files:
            print('>>> Train on channel: {0}'.format(f))
            data_matrix = []

            topic_test_yhat = train_with_topics(os.path.join(subdir, f))

            with open(os.path.join(subdir, f), 'r') as fin:
                for line in fin:
                    vid, duration, categoryId, topics, watch_prec = line.rstrip().split('\t')
                    # topics = topics.split(',')
                    # cnt = 0
                    # sum_xy = 0
                    # for topic in topics:
                    #     cnt += 1
                    #     sum_xy += topic_regression.coef_[1+topic_lookup[topic]]
                    # video_topics_length = np.sqrt(cnt)
                    # # print(topics)
                    # # # print(channel_topics)
                    # # print(sum_xy)
                    # # print(video_topics_length)
                    # # print(channel_topics_length)
                    # # print(float(sum_xy/video_topics_length/channel_topics_length))
                    # data_matrix.append(np.array([int(duration), float(sum_xy/video_topics_length/channel_topics_length), float(watch_prec)]))
                    data_matrix.append(np.array([int(duration), float(watch_prec)]))

            df = pd.DataFrame(np.array(data_matrix), columns=['video duration', 'watch percentage'])
            knn_regression(df, fig_idx, f, topic_test_yhat)
            fig_idx += 1
            print('-'*79)
            print()

    # ax1 = fig.add_subplot(111)
    # evaluation_matrix = [global_errors, knn_errors, linreg_errors, logreg_errors]
    # print([len(x) for x in evaluation_matrix])
    # ax1.boxplot(evaluation_matrix, labels=['MLE', 'KNN', 'LinReg', 'LinReg-Log'], showfliers=False, showmeans=True)
    # ax1.set_ylabel('mean absolute error')
    #
    # means = [np.mean(x) for x in evaluation_matrix]
    # means_labels = ['{0:.4f}%'.format(s*100) for s in means]
    # pos = range(len(means))
    # for tick, label in zip(pos, ax1.get_xticklabels()):
    #     ax1.text(pos[tick] + 1, means[tick] + 0.01, means_labels[tick], horizontalalignment='center', size='medium', color='k')

    # plt.tight_layout()
    # plt.show()
