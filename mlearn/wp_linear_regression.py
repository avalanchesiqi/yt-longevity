from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset = pd.read_csv('vevo_data_regression.txt', delimiter=',', dtype='float32', header=None)
    # remove nan instance
    dataset = dataset.dropna()
    print "Data loading Finished!"

    # topic feature
    features = dataset.iloc[:, :-5]
    # duration feature
    durations = dataset.iloc[:, -5].values.reshape(-1, 1)
    dur_std_scale = preprocessing.StandardScaler().fit(durations)
    dur_std = dur_std_scale.transform(durations)
    features.loc[:, 2079] = dur_std
    # definition feature
    definitions = dataset.iloc[:, -4].values.reshape(-1, 1)
    features.loc[:, 2080] = definitions
    # date feature
    dates = dataset.iloc[:, -3].values.reshape(-1, 1)
    date_minmax_scale = preprocessing.MinMaxScaler().fit(dates)
    date_std = date_minmax_scale.transform(dates)
    features.loc[:, 2081] = date_std
    # past average watch time feature
    p_avg_wt = dataset.iloc[:, -2].values.reshape(-1, 1)
    features.loc[:, 2082] = p_avg_wt

    labels = dataset.iloc[:, -1]
    labels[labels < 0] = 0
    labels[labels > 100] = 100

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)
    print "Data split Finished!"

    lr = linear_model.LinearRegression(n_jobs=-1)
    lr.fit(train_features, train_labels)

    train_predicted = lr.predict(train_features)
    train_predicted[train_predicted < 0] = 0
    train_predicted[train_predicted > 100] = 100

    test_predicted = lr.predict(test_features)
    test_predicted[test_predicted < 0] = 0
    test_predicted[test_predicted > 100] = 100

    for pair in zip(test_predicted[-300:-100], test_labels.iloc[-300:-100].values):
        print pair

    print '----------------------'
    print 'train rmse:', np.sqrt(metrics.mean_squared_error(train_labels, train_predicted))
    print 'test rmse:', np.sqrt(metrics.mean_squared_error(test_labels, test_predicted))
    print 'benchmark rmse:', np.sqrt(metrics.mean_squared_error(test_labels, [np.mean(train_labels)]*len(test_labels)))

    # plt.hist(test_labels, alpha=0.5, color='g', bins=range(100))
    # plt.hist(test_predicted, alpha=0.5, color='r', bins=range(100))
    plt.scatter(test_labels, test_predicted, s=5, facecolors='none')
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.xlabel('True')
    plt.ylabel('Predict')
    plt.axes().set_aspect('equal')

    plt.show()
