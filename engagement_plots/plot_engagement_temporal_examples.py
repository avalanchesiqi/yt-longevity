#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Scripts to plot relative engagement temporal fitting examples.
video 1: HKK99xm--Ro
video 2: rKdNjlNYMKk

Usage: python plot_engagement_temporal_examples.py
Time: ~2M
"""

from __future__ import division, print_function
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from scipy import optimize
import cPickle as pickle
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

from utils.helper import read_as_float_array, read_as_int_array
from utils.converter import to_relative_engagement


def predict(params, x):
    c, lam = params
    x_predict = None
    for i in x:
        if i == 0:
            x_predict = np.array([c])
        else:
            curr_predict = np.array([c*(i**lam)])
            x_predict = np.concatenate([x_predict, curr_predict], axis=0)
    return x_predict


def cost_function(params, x, y):
    re_predict = predict(params, x)
    cost_vector = re_predict - y
    cost = np.sum(abs(cost_vector))
    return cost/len(cost_vector)


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    age = 120
    autograd_func = grad(cost_function)

    engagement_map_loc = '../data_preprocess/engagement_map.p'
    if not os.path.exists(engagement_map_loc):
        print('Engagement map not generated, start with generating engagement map first in ../data_preprocess dir!.')
        print('Exit program...')
        sys.exit(1)

    engagement_map = pickle.load(open(engagement_map_loc, 'rb'))
    lookup_durations = np.array(engagement_map['duration'])

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    data_loc = '../../production_data/new_tweeted_dataset_norm/test_data'
    fig = plt.figure(figsize=(9, 9))

    for subdir, _, files in os.walk(data_loc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                fin.readline()
                for line in fin:
                    dump, days, daily_view, daily_watch = line.rstrip().rsplit('\t', 3)
                    vid, _, duration, _ = dump.split('\t', 3)
                    if vid == 'HKK99xm--Ro':
                        fig_idx = 0
                    elif vid == 'rKdNjlNYMKk':
                        fig_idx = 1
                    else:
                        continue
                    duration = int(duration)
                    days = read_as_int_array(days, delimiter=',', truncated=age)
                    daily_view = read_as_int_array(daily_view, delimiter=',', truncated=age)
                    daily_watch = read_as_float_array(daily_watch, delimiter=',', truncated=age)

                    cumulative_wp = [0]*age
                    for i in range(age):
                        cumulative_wp[i] = np.sum(daily_watch[days <= i]) * 60 / np.sum(daily_view[days <= i]) / duration

                    cumulative_engagement = to_relative_engagement(engagement_map, duration, cumulative_wp, lookup_keys=lookup_durations)

                    optimizer = optimize.minimize(cost_function, np.array([0.5, 1]), jac=autograd_func,
                                                  method='L-BFGS-B', args=(np.arange(120), cumulative_engagement),
                                                  bounds=[(None, None), (None, None)], options={'disp': None})

                    # == == == == == == == == Part 3: Plot fitting result == == == == == == == == #
                    to_plot = True
                    if to_plot:
                        ax1 = fig.add_subplot(2, 1, 1+fig_idx)
                        ax2 = ax1.twinx()
                        ax1.plot(days, daily_view, 'b-')
                        ax2.plot(days, cumulative_engagement, 'k--')
                        ax2.plot(days, predict(optimizer.x, days), 'r-')

                        if fig_idx == 0:
                            ax1.set_ylabel(r'daily view $x_v$', color='b', fontsize=24)
                            ax2.set_ylabel(r'relative engagement $\bar \eta_{t}$', color='k', fontsize=24)
                            for ax in [ax1, ax2]:
                                for label in ax.get_xticklabels()[:]:
                                    label.set_visible(False)
                        elif fig_idx == 1:
                            ax1.set_xlabel('video age (day)', fontsize=24)

                        ax1.set_xlim([0, 125])
                        ax1.set_ylim(ymin=max(0, ax1.get_ylim()[0]))
                        ax2.set_ylim([0, 1])
                        ax1.tick_params('y', colors='b')
                        ax2.tick_params('y', colors='k')

                        annotated_str = r'ID: {0}'.format(vid)
                        annotated_str += '\n'
                        annotated_str += r'$C$: {0:.4f}, $\lambda$: {1:.4f}'.format(*optimizer.x)
                        ax2.text(120, 0.77, annotated_str, horizontalalignment='right', fontsize=24)

                        ax2.set_xticks([0, 40, 80, 120])
                        display_min = int(np.floor(min(daily_view) / 100) * 100)
                        display_max = int(np.ceil(max(daily_view) / 100) * 100)
                        ax1.set_yticks([display_min, (display_min+display_max)/2, display_max])
                        ax2.set_yticks([0.0, 0.5, 1.0])
                        for ax in [ax1, ax2]:
                            plt.setp(ax.yaxis.get_majorticklabels(), rotation=90)
                            ax.tick_params(axis='both', which='major', labelsize=24)

    plt.legend([plt.Line2D((0, 1), (0, 0), color='k', linestyle='--'),
                plt.Line2D((0, 1), (0, 0), color='b'), plt.Line2D((0, 1), (0, 0), color='r')],
               ['Observed relative engagement', 'Observed watch time', 'Fitted relative engagement'],
               fontsize=18, frameon=False, handlelength=1,
               loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)

    plt.tight_layout(rect=[0, 0.08, 1, 1], h_pad=0)
    plt.show()
