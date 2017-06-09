from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

if __name__ == '__main__':
    dataset = pd.read_csv('vevo_data_regression.txt', delimiter='\t', dtype='float32')
    # remove nan instance
    dataset = dataset.dropna()
    print "Data loading Finished!"

    labels = dataset.iloc[:, -1]
    labels[labels < 0] = 0
    labels[labels > 100] = 100

    # topic feature
    features = dataset.iloc[:, :-5]
    # duration feature
    durations = dataset.iloc[:, -5].values.reshape(-1, 1)
    dur_std_scale = preprocessing.StandardScaler().fit(durations)
    dur_std = dur_std_scale.transform(durations)
    print 'pearson duration vs wp:', stats.pearsonr(dur_std, labels.values.reshape(-1, 1))
    features.loc[:, 'duration'] = dur_std
    # definition feature
    definitions = dataset.iloc[:, -4].values.reshape(-1, 1)
    features.loc[:, 'definition'] = definitions
    # date feature
    dates = dataset.iloc[:, -3].values.reshape(-1, 1)
    date_minmax_scale = preprocessing.MinMaxScaler().fit(dates)
    date_std = date_minmax_scale.transform(dates)
    print 'pearson age vs wp:', stats.pearsonr(date_std, labels.values.reshape(-1, 1))
    features.loc[:, 'date'] = date_std
    # past average watch time feature
    p_avg_wt = dataset.iloc[:, -2].values.reshape(-1, 1)
    pwt_std_scale = preprocessing.StandardScaler().fit(p_avg_wt)
    pwt_std = pwt_std_scale.transform(p_avg_wt)
    print 'pearson past success vs wp:', stats.pearsonr(pwt_std, labels.values.reshape(-1, 1))
    features.loc[:, 'past success'] = p_avg_wt

    # features = sm.add_constant(features)

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)
    print "Data split Finished!"

    lr = linear_model.Ridge()
    lr.fit(train_features, train_labels)
    # model = sm.OLS(train_labels, train_features)
    # results = model.fit()
    # print(results.summary())

    # train_predicted = lr.predict(train_features)
    # train_predicted[train_predicted < 0] = 0
    # train_predicted[train_predicted > 100] = 100
    #
    test_predicted = lr.predict(test_features)
    # test_predicted = results.predict(test_features)
    test_predicted[test_predicted < 0] = 0
    test_predicted[test_predicted > 100] = 100
    #
    # for pair in zip(test_predicted[-500:-400], test_labels.iloc[-500:-400].values):
    #     print pair

    col_name = dataset.columns.tolist()
    for i, j in enumerate(lr.coef_):
        print '{0} {1}'.format(col_name[i].ljust(50), j)

    print 'intercept', lr.intercept_

    print '----------------------'
    absolute_error = np.abs(test_labels.values - test_predicted)
    relative_absolute_error = np.abs(test_labels.values - test_predicted)/test_labels.values
    print 'Accuracy: {0:.2f} (+/- {1:.2f})'.format(np.mean(absolute_error), np.std(absolute_error)*2)
    print 'Relative accuracy: {0:.2f} (+/- {1:.2f})'.format(np.mean(relative_absolute_error), np.std(relative_absolute_error)*2)
    # print 'train rmse:', np.sqrt(metrics.mean_squared_error(train_labels, train_predicted))
    # print 'test rmse:', np.sqrt(metrics.mean_squared_error(test_labels, test_predicted))
    # print 'benchmark rmse:', np.sqrt(metrics.mean_squared_error(test_labels, [np.mean(train_labels)]*len(test_labels)))
    # print 'train mae:', metrics.mean_absolute_error(train_labels, train_predicted)
    benchmark_mae = np.abs(test_labels - np.array([np.mean(train_labels)] * len(test_labels)))
    print 'test mae five points summary:', np.mean(absolute_error), np.max(absolute_error), np.percentile(absolute_error, 75), np.median(absolute_error), np.percentile(absolute_error, 25), np.min(absolute_error)
    print 'benchmark mae:', np.mean(benchmark_mae), np.max(benchmark_mae), np.percentile(benchmark_mae, 75), np.median(benchmark_mae), np.percentile(benchmark_mae, 25), np.min(benchmark_mae)

    # plt.hist(test_labels, alpha=0.5, color='g', bins=range(100))
    # plt.hist(test_predicted, alpha=0.5, color='r', bins=range(100))
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # plt.boxplot(absolute_error, showmeans=True, showfliers=False)
    # plt.boxplot(np.abs(test_labels.values - np.array([np.mean(train_labels)] * len(test_labels))), showmeans=True, showfliers=False)
    # plt.xlim([0, 100])
    # plt.ylim([0, 100])
    # plt.xlabel('True')
    # plt.ylabel('Predict')
    # plt.axes().set_aspect('equal')

    # plt.show()
