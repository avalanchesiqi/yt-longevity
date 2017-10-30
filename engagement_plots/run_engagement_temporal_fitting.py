#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Scripts to run relative engagement temporal fitting.

Usage: python run_engagement_temporal_fitting.py
Time: 
"""

from __future__ import division, print_function
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from scipy import optimize
import cPickle as pickle
import autograd.numpy as np
from autograd import grad

from utils.helper import read_as_float_array, read_as_int_array, strify
from utils.converter import to_relative_engagement


# power-law method
def predict1(params, x):
    c, lam = params
    x_predict = None
    for i in x:
        if i == 0:
            x_predict = np.array([c])
        else:
            curr_predict = np.array([c*(i**lam)])
            x_predict = np.concatenate([x_predict, curr_predict], axis=0)
    return x_predict


def cost_function1(params, x, y):
    re_predict = predict1(params, x)
    cost_vector = re_predict - y
    cost = np.sum(abs(cost_vector))
    return cost/len(cost_vector)


# linear method
def predict2(params, x):
    w, b = params
    x_predict = None
    for i in x:
        if i == 0:
            x_predict = np.array([b])
        else:
            curr_predict = np.array([w*i + b])
            x_predict = np.concatenate([x_predict, curr_predict], axis=0)
    return x_predict


def cost_function2(params, x, y):
    re_predict = predict2(params, x)
    cost_vector = re_predict - y
    cost = np.sum(abs(cost_vector))
    return cost/len(cost_vector)


# constant method
def predict3(params, x):
    const = params
    x_predict = np.array([const] * len(x)).reshape(len(x), )
    return x_predict


def cost_function3(params, x, y):
    re_predict = predict3(params, x)
    cost_vector = re_predict - y
    cost = np.sum(abs(cost_vector))
    return cost/len(cost_vector)


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    age = 120
    window_size = 14
    min_window_view = 100
    autograd_func1 = grad(cost_function1)
    autograd_func2 = grad(cost_function2)
    autograd_func3 = grad(cost_function3)

    engagement_map_loc = '../data_preprocess/engagement_map.p'
    if not os.path.exists(engagement_map_loc):
        print('Engagement map not generated, start with generating engagement map first in ../data_preprocess dir!.')
        print('Exit program...')
        sys.exit(1)

    engagement_map = pickle.load(open(engagement_map_loc, 'rb'))
    lookup_durations = np.array(engagement_map['duration'])

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    data_loc = '../../production_data/new_tweeted_dataset_norm/'
    log_path = 'temporal_fitting.txt'
    log_data = open(log_path, 'w')

    for subdir, _, files in os.walk(data_loc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                fin.readline()
                for line in fin:
                    dump, days, daily_view, daily_watch = line.rstrip().rsplit('\t', 3)
                    vid, _, duration, _ = dump.split('\t', 3)
                    duration = int(duration)
                    days = read_as_int_array(days, delimiter=',', truncated=age)
                    daily_view = read_as_int_array(daily_view, delimiter=',', truncated=age)
                    daily_watch = read_as_float_array(daily_watch, delimiter=',', truncated=age)

                    # a moving windows solution, using past 7 days to calculate wp
                    cumulative_wp = []
                    for i in range(days[-1] + 1):
                        if i < window_size:
                            past_window_views = np.sum(daily_view[days <= i])
                            past_window_watches = np.sum(daily_watch[days <= i])
                        else:
                            past_window_views = np.sum(daily_view[(i - window_size < days) & (days <= i)])
                            past_window_watches = np.sum(daily_watch[(i - window_size < days) & (days <= i)])
                        if past_window_views < min_window_view:
                            break
                        cumulative_wp.append(past_window_watches * 60 / past_window_views / duration)

                    if len(cumulative_wp) > 60:
                        cumulative_engagement = to_relative_engagement(engagement_map, duration, cumulative_wp, lookup_keys=lookup_durations)
                        truncated_age = len(cumulative_engagement)

                        optimizer1 = optimize.minimize(cost_function1, np.array([0.5, 1]), jac=autograd_func1,
                                                       method='L-BFGS-B',
                                                       args=(np.arange(truncated_age), cumulative_engagement),
                                                       bounds=[(None, None), (None, None)], options={'disp': None})

                        optimizer2 = optimize.minimize(cost_function2, np.array([0.5, 1]), jac=autograd_func2,
                                                       method='L-BFGS-B',
                                                       args=(np.arange(truncated_age), cumulative_engagement),
                                                       bounds=[(None, None), (None, None)], options={'disp': None})

                        optimizer3 = optimize.minimize(cost_function3, np.array([0.5]), jac=autograd_func3,
                                                       method='L-BFGS-B',
                                                       args=(np.arange(truncated_age), cumulative_engagement),
                                                       bounds=[(0, 1)], options={'disp': None})

                        log_data.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\n'
                                       .format(vid, truncated_age, optimizer1.fun, optimizer2.fun, optimizer3.fun,
                                               strify(optimizer1.x), strify(optimizer2.x), strify(optimizer3.x)))
                        print(vid, truncated_age, optimizer1.fun, optimizer2.fun, optimizer3.fun,
                              strify(optimizer1.x), strify(optimizer2.x), strify(optimizer3.x))
    log_data.close()
