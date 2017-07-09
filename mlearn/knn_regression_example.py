from __future__ import print_function, division
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection, neighbors, metrics, linear_model


def knn_regression(data, idx):
    train_size = 0.8
    train_data = data[:int(train_size*len(data))]
    test_data = data[int(train_size*len(data)):]

    train_x = train_data['video duration'].reshape(-1, 1)
    test_x = test_data['video duration'].reshape(-1, 1)
    train_y = train_data['watch percentage'].reshape(-1, 1)
    test_y = test_data['watch percentage'].reshape(-1, 1)

    T = np.linspace(np.min(train_x, 0), np.max(train_x, 0), 250)[:, np.newaxis]

    # kf = model_selection.KFold(n_splits=5)
    #
    # # select best neighbors number
    # best_err = np.inf
    # best_neighbors_num = None
    # m = int(0.8*duration.shape[0])
    # for n_neighbors in xrange(1, m):
    #     error_list = []
    #     for train_index, test_index in kf.split(duration):
    #         X_train, X_test = duration[train_index], duration[test_index]
    #         y_train, y_test = wp[train_index], wp[test_index]
    #
    #         knn_initializer = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    #         knn_model = knn_initializer.fit(X_train, y_train)
    #         y_hat = knn_model.predict(X_test)
    #         error_list.append(metrics.mean_absolute_error(y_test, y_hat))
    #     tot_err = np.sum(error_list)
    #     if tot_err < best_err:
    #         best_err = tot_err
    #         best_neighbors_num = n_neighbors

    print('>>> train: {0} videos; test: {1} videos'.format(len(train_x), len(test_x)))

    # knn model
    n_neighbors = 5
    weights = 'distance'
    knn_initializer = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    knn_model = knn_initializer.fit(train_x, train_y)
    y_ = knn_model.predict(T)
    print('>>> mean absolute error knn: {0:.4f}'.format(metrics.mean_absolute_error(test_y, knn_model.predict(test_x))))

    # linear regression model
    linreg = linear_model.Ridge().fit(train_x, train_y)
    y__ = linreg.predict(T)
    print('>>> mean absolute error ridge: {0:.4f}'.format(metrics.mean_absolute_error(test_y, linreg.predict(test_x))))

    ax1 = fig.add_subplot(221 + idx)
    ax1.scatter(train_x, train_y, c='k', s=10, label='train data')
    ax1.plot(T, y_, c='g', label='knn@{0}'.format(n_neighbors))
    ax1.plot(T, y__, '--', c='b', label='linreg')
    ax1.scatter(test_x, test_y, c='r', s=10, marker='x', label='test data')
    ax1.set_xlabel('Video Duration')
    ax1.set_ylabel('Watch Percentage')
    ax1.set_xlim(xmin=0)
    ax1.set_ylim([0, 1])
    plt.legend(loc='best')
    # plt.title('KNNeighborsRegressor (k = {0}, weights = {1})'.format(best_neighbors_num, weights))
    print('-'*79)


if __name__ == '__main__':
    fig = plt.figure(figsize=(10, 8))

    input_doc = '../../data/production_data/active_channels_feature'

    cnt = 0
    for subdir, _, files in os.walk(input_doc):
        for f in files:
            data_matrix = []
            with open(os.path.join(subdir, f), 'r') as fin:
                for line in fin:
                    vid, duration, categoryId, topics, watch_prec = line.rstrip().split('\t')
                    data_matrix.append(np.array([int(duration), float(watch_prec)]))
            df = pd.DataFrame(np.array(data_matrix), columns=['video duration', 'watch percentage'])
            knn_regression(df, cnt)
            cnt += 1
            if cnt > 3:
                break

    plt.tight_layout()
    plt.show()
