#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
from sklearn import metrics, neighbors, linear_model, model_selection
import matplotlib.pyplot as plt

# KNN regression for watch percentage prediction as baseline model
# Independent variable: video duration
# Dependent variable: watch percentage in 1st week


def report_summary(arr):
    print '+++++++++++++++++++++++++++'
    print '+ Mean         {0:>10.3f} +'.format(np.mean(arr))
    print '+ Max          {0:>10.3f} +'.format(np.max(arr))
    print '+ 75 percentile{0:>10.3f} +'.format(np.percentile(arr, 75))
    print '+ Median       {0:>10.3f} +'.format(np.median(arr))
    print '+ 25 percentile{0:>10.3f} +'.format(np.percentile(arr, 25))
    print '+ Min          {0:>10.3f} +'.format(np.min(arr))
    print '+++++++++++++++++++++++++++'


def main():
    # == == == == == = Part 1: Loading Data == == == == == == = #
    input_path = 'vevo_past_success.txt'
    data = None
    with open(input_path, 'r') as f:
        for line in f:
            tripets, channel_id = line.rstrip().rsplit(None, 1)
            duration, _, wp = np.array(map(lambda x: x.split(','), tripets.split())).T
            duration = duration.astype('int').reshape(-1, 1)
            wp = wp.astype('float').reshape(-1, 1)
            m = len(duration)
            extend_data = np.hstack((np.full((m, 1), channel_id, dtype=object), duration, wp))
            if data is None:
                data = extend_data
            else:
                data = np.vstack((data, extend_data))

    print 'Finish loading data.'
    print 'Create a numpy matrix with shape {0} x {1}.'.format(data.shape[0], data.shape[1])

    # == == == == == = Part 2: Splitting Data == == == == == == = #
    data[data[:, 2]>100] = 100
    data_train, data_test = model_selection.train_test_split(data, test_size=0.2, random_state=25)
    X_train = data_train[:, 1].reshape(-1, 1)
    y_train = data_train[:, 2].reshape(-1, 1)
    X_test = data_test[:, 1].reshape(-1, 1)
    y_test = data_test[:, 2].reshape(-1, 1)

    # == == == == == = Part 3: Train a global model with KNN and Ridge Reg == == == == == == = #
    # knn_mae = []
    # weights = 'distance'
    # neighbors_search_range = np.arange(1, 400)
    # for n_neighbors in neighbors_search_range:
    #     knn_initializer = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    #     knn_model = knn_initializer.fit(X_train, y_train)
    #     y_pred = knn_model.predict(X_test)
    #     knn_mae.append(metrics.mean_absolute_error(y_test, y_pred))
    #
    # knn_mae = np.array(knn_mae)
    # best_neighbors_num = neighbors_search_range[np.argmin(knn_mae)]
    #
    # plt.subplot(2, 1, 1)
    # plt.plot(neighbors_search_range, knn_mae)

    weights = 'distance'
    best_neighbors_num = 300
    knn_initializer = neighbors.KNeighborsRegressor(best_neighbors_num, weights=weights)
    global_knn_model = knn_initializer.fit(X_train, y_train)
    y_pred = global_knn_model.predict(X_test)
    print 'global knn model'
    report_summary(np.abs(y_test-y_pred))

    # ridge_mae = []
    # alpha_search_range = np.arange(0.01, 100, 0.01)
    # for alpha in alpha_search_range:
    #     ridge_initializer = linear_model.Ridge(alpha=alpha)
    #     ridge_model = ridge_initializer.fit(X_train, y_train)
    #     y_pred = ridge_model.predict(X_test)
    #     ridge_mae.append(metrics.mean_absolute_error(y_test, y_pred))
    #
    # ridge_mae = np.array(ridge_mae)
    # best_alpha_num = alpha_search_range[np.argmin(ridge_mae)]
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(alpha_search_range, ridge_mae)

    best_alpha_num = 0.01
    ridge_initializer = linear_model.Ridge(alpha=best_alpha_num)
    global_ridge_model = ridge_initializer.fit(X_train, y_train)
    y_pred = global_ridge_model.predict(X_test)
    print 'global ridge model'
    report_summary(np.abs(y_test-y_pred))

    # plt.show()

    # == == == == == = Part 4: Train per-user models with KNN and Ridge Reg == == == == == == = #
    uniq_channels = np.unique(data_train[:, 0])
    channel_knn_dict = {}
    channel_ridge_dict = {}
    for channel_id in uniq_channels:
        channel_idx = (data_train[:, 0] == channel_id)
        channel_X_train = X_train[channel_idx]
        channel_y_train = y_train[channel_idx]

        n_neighbors = channel_X_train.shape[0]
        weights = 'distance'
        knn_initializer = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
        channel_knn_model = knn_initializer.fit(channel_X_train, channel_y_train)
        channel_knn_dict[channel_id] = channel_knn_model

        alpha = 0.01
        ridge_initializer = linear_model.Ridge(alpha=alpha)
        channel_ridge_model = ridge_initializer.fit(channel_X_train, channel_y_train)
        channel_ridge_dict[channel_id] = channel_ridge_model

    # == == == == == = Part 5: Predict if channel model presence or global model == == == == == == = #
    uniq_test_channels = np.unique(data_test[:, 0])
    knn_y_pred = np.zeros((data_test.shape[0], 1))
    ridge_y_pred = np.zeros((data_test.shape[0], 1))
    for channel_id in uniq_test_channels:
        channel_idx = (data_test[:, 0] == channel_id)
        channel_X_test = X_test[channel_idx]
        if channel_id in channel_knn_dict:
            # per channel model
            channel_knn_model = channel_knn_dict[channel_id]
            channel_y_pred = channel_knn_model.predict(channel_X_test)
            knn_y_pred[channel_idx] = channel_y_pred

            channel_ridge_model = channel_ridge_dict[channel_id]
            channel_y_pred = channel_ridge_model.predict(channel_X_test)
            ridge_y_pred[channel_idx] = channel_y_pred
        else:
            # global model
            global_y_pred = global_knn_model.predict(channel_X_test)
            knn_y_pred[channel_idx] = global_y_pred

            global_y_pred = global_ridge_model.predict(channel_X_test)
            ridge_y_pred[channel_idx] = global_y_pred

    # == == == == == = Part 6: Evaluate measurement metric == == == == == == = #
    print 'per channel knn model'
    report_summary(np.abs(y_test-knn_y_pred))
    print 'per channel ridge model'
    report_summary(np.abs(y_test-ridge_y_pred))


if __name__ == '__main__':
    main()
