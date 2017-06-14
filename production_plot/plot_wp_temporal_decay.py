#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import os
import isodate
from datetime import timedelta
from scipy import optimize, stats
from sklearn import linear_model, metrics
import autograd.numpy as np
from autograd import grad
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Plot watch percentage temporal decay
# Usage: python plot_wp_temporal_decay.py /Volumes/mbp/Users/siqi/OData/info_43.txt /Volumes/mbp/Users/siqi/OData/wp_plot /Volumes/mbp/Users/siqi/OData/log/shows.log

category_dict = {"42": "Shorts", "29": "Nonprofits & Activism", "24": "Entertainment", "25": "News & Politics", "26": "Howto & Style", "27": "Education", "20": "Gaming", "21": "Videoblogging", "22": "People & Blogs", "23": "Comedy", "44": "Trailers", "28": "Science & Technology", "43": "Shows", "40": "Sci-Fi/Fantasy", "41": "Thriller", "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music", "39": "Horror", "38": "Foreign", "15": "Pets & Animals", "17": "Sports", "19": "Travel & Events", "18": "Short Movies", "31": "Anime/Animation", "30": "Movies", "37": "Family", "36": "Drama", "35": "Documentary", "34": "Comedy", "33": "Classics", "32": "Action/Adventure"}


def read_as_int_array(content, truncated=None, delimiter=None):
    """
    Read input as an int array.
    :param content: string input
    :param truncated: head number of elements extracted
    :param delimiter: delimiter string
    :return: a numpy int array
    """
    if truncated is None:
        return np.array(map(int, content.split(delimiter)), dtype=np.uint32)
    else:
        return np.array(map(int, content.split(delimiter, truncated)[:-1]), dtype=np.uint32)


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
        return np.array(map(float, content.split(delimiter, truncated)[:-1]), dtype=np.float64)


def predict(params, x_input):
    """
    Predict property value via an exponential decaying.
    :param params: scale factor mu and decay factor theta
    :param x_input: time series, from 0 or 1 to 180
    :return: predict property value
    """
    mu, theta = params
    x_predict = None
    for idx, t in enumerate(x_input):
        if idx == 0:
            x_predict = np.array([mu])
        else:
            curr_predict = np.array([mu*(t**(-theta))])
            x_predict = np.concatenate([x_predict, curr_predict], axis=0)
    return x_predict


def cost_function(params, x_input, y_true, const):
    """
    MSE of predict value and ground-true value
    :param params: scale factor mu and decay factor theta
    :param x_input: time series, from 0 or 1 to 180
    :param y_true: ground-true watch percentage
    :param const: whether fitted by a constant value or not
    :return: cost function in MSE format
    """
    if const:
        y_predict = np.array([params[0]]*len(y_true))
    else:
        y_predict = predict(params, x_input)
    cost_vector = y_predict - y_true
    total_cost = np.sum(cost_vector ** 2) / 2
    return total_cost/len(cost_vector)


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    fig = plt.figure()
    age = 180
    autograd_func = grad(cost_function)
    logging.basicConfig(filename=sys.argv[3], level=logging.DEBUG)

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    data_loc = sys.argv[1]
    output_loc = sys.argv[2]
    # make output dir if not exists
    if not os.path.exists(output_loc):
        os.makedirs(output_loc)

    with open(data_loc) as fin:
        # id, duration, definition,
        # categoryId, channelId, publishedAt,
        # titleLength, titlePolarity, descLength, descPolarity,
        # topics, topicsNum,
        # days, dailyView, dailyWatch
        fin.readline()
        for line in fin:
            id, duration, definition, category_id, channel_id, published_at, len_title, polar_title, len_desc, polar_desc, topics, topics_num, days, daily_view, daily_watch = line.rstrip().split('\t')
            duration = int(duration)
            duration_iso = isodate.duration_isoformat(timedelta(seconds=duration))

            days = read_as_int_array(days, truncated=age, delimiter=',')
            if len(days) < 60:
                continue

            daily_view = read_as_int_array(daily_view, truncated=len(days), delimiter=',')

            # pre filter
            if len(days[daily_view < 20]) > 1/4*len(days):
            # if sum(daily_view[days < 180]) < 1000:
                continue

            daily_watch = read_as_float_array(daily_watch, truncated=len(days), delimiter=',')

            daily_wp = np.divide(daily_watch * 60, daily_view * duration, where=(daily_view != 0))
            daily_wp[daily_wp > 1] = 1

            # fitted by exponential decay
            init_mu = 0.5
            init_theta = 0
            optimizer = optimize.minimize(cost_function, np.array([init_mu, init_theta]), jac=autograd_func, method='L-BFGS-B',
                                          args=(days, daily_wp, False), bounds=[(0, 1), (None, None)],
                                          options={'disp': None})

            # fitted by ridge regression
            n = len(days)
            lr = linear_model.Ridge()
            train_features = days.reshape(n, 1)
            train_labels = daily_wp.reshape(n, 1)
            lr.fit(train_features, train_labels)

            # fitted by a constant number
            optimizer2 = optimize.minimize(cost_function, np.array([init_mu, 0]), jac=autograd_func, method='L-BFGS-B',
                                           args=(days, daily_wp, True), bounds=[(0, 1), (None, None)],
                                           options={'disp': None})

            logging.debug('exp: {0}, ridge: {1}, constant: {2}'
                          .format(np.sqrt(optimizer.fun),
                                  np.sqrt(metrics.mean_squared_error(lr.predict(train_features), train_labels)),
                                  np.sqrt(optimizer2.fun)))

            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()
            ax1.plot(days, daily_view, 'r-', label='observed #views')
            ax2.plot(days, daily_wp, '--', c='k', ms=2, mfc='None', mec='k', mew=1, label='daily watch percentage')
            ax2.plot(days, predict(optimizer.x, days), 'o-', c='g', ms=2, mfc='None', mec='g', mew=1, label='fitted watch percentage')

            ax1.set_xlim(xmax=180)
            ax1.set_ylim(ymin=max(0, ax1.get_ylim()[0]))
            ax2.set_xlim(xmax=180)
            ax2.set_ylim(ymin=0)
            ax2.set_ylim(ymax=1)
            ax1.set_xlabel('video age (day)')
            ax1.set_ylabel('Number of views', color='r')
            ax1.tick_params('y', colors='r')
            ax2.set_ylabel('Portion of watch', color='k')
            ax2.tick_params('y', colors='k')

            ax2.text(2, 0.83, '{0}\n{1}, {2}\nmu: {3:.4f}, theta: {4:.4f}'
                     .format(id, category_dict[category_id], duration_iso, *optimizer.x), bbox={'facecolor': 'green', 'alpha': 0.5})

            fig.savefig(os.path.join(output_loc, id))
            plt.clf()
