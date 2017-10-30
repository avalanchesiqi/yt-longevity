#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Scripts to plot Figure 6, plot regressor prediction results for watch percentage."""

from __future__ import division, print_function
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import cPickle as pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

from utils.converter import to_watch_percentage


def plot_barchart(mae_list, r2_list):
    n = len(mae_list) // 2
    raw_mae = [mae_list[2*i] for i in range(n)]
    converted_mae = [mae_list[2*i+1] for i in range(n)]
    raw_r2 = [r2_list[2*i] for i in range(n)]
    converted_r2 = [r2_list[2*i+1] for i in range(n)]

    # generate barchart plot
    fig = plt.figure(figsize=(8, 3))
    width = 0.4
    n = len(raw_mae)
    ind = np.arange(n)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    cornflower_blue = '#6495ed'
    tomato = '#ff6347'

    ax1.bar(ind - width / 2, raw_mae, width, edgecolor=['k'] * n, color=cornflower_blue, lw=1.5)
    ax1.bar(ind + width / 2, converted_mae, width, color='none', edgecolor=[cornflower_blue]*n, hatch='//', lw=1.5)

    ax2.bar(ind - width / 2, raw_r2, width, edgecolor=['k'] * n, color=tomato, lw=1.5)
    ax2.bar(ind + width / 2, converted_r2, width, color='none', edgecolor=[tomato]*n, hatch='\\\\', lw=1.5)

    ax1.set_ylim([0, 0.21])
    ax2.set_ylim([0, 1])
    ax1.set_yticks([0, 0.1, 0.2])
    ax2.set_yticks([0.2, 0.6, 1.0])
    ax1.set_ylabel(r'$MAE$', fontsize=16)
    ax2.set_ylabel(r'$R^2$', fontsize=16)

    for label in ax1.get_xticklabels()[:]:
        label.set_visible(False)
    ax2.set_xticklabels(('', 'D', 'C', 'T', 'C+T', 'R', 'All', 'CSP'))

    for tick, label in zip(ind, ax1.get_xticklabels()):
        ax1.text(ind[tick]-width/2-0.05, raw_mae[tick]+0.01, [str(np.round(x, 3)) for x in raw_mae][tick],
                 horizontalalignment='center', color='k', fontsize=12)
        ax1.text(ind[tick]+width/2+0.05, converted_mae[tick]+0.01, [str(np.round(x, 3)) for x in converted_mae][tick],
                 horizontalalignment='center', color=cornflower_blue, fontsize=12)
    for tick, label in zip(ind, ax2.get_xticklabels()):
        ax2.text(ind[tick]-width/2-0.05, raw_r2[tick]+0.01, [str(np.round(x, 3)) for x in raw_r2][tick],
                 horizontalalignment='center', color='k', fontsize=12)
        ax2.text(ind[tick]+width/2+0.05, converted_r2[tick]+0.01, [str(np.round(x, 3)) for x in converted_r2][tick],
                 horizontalalignment='center', color=tomato, fontsize=12)

    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    ax1.set_title(r'(b) Predict watch percentage $\bar \mu_{30}$', fontsize=18)

    ax1.legend([plt.Rectangle((0, 0), 1, 1, fc=cornflower_blue, ec='k'),
                plt.Rectangle((0, 0), 1, 1, fc='none', ec=cornflower_blue, hatch='//'),
                plt.Rectangle((0, 0), 1, 1, fc=tomato, ec='k'),
                plt.Rectangle((0, 0), 1, 1, fc='none', ec=tomato, hatch='\\\\')],
               ['MAE', r'MAE via $\bar \eta_{30}$', r'$R^2$', r'$R^2$ via $\bar \eta_{30}$'], fontsize=14,
               frameon=False, loc='lower center', bbox_to_anchor=(0.5, -2.3), fancybox=False, shadow=True, ncol=4)

    plt.tight_layout(rect=[0.01, 0.05, 1, 0.99])
    plt.show()


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    engagement_map_loc = '../data_preprocess/engagement_map.p'
    if not os.path.exists(engagement_map_loc):
        print('Engagement map not generated, start with generating engagement map first in ../data_preprocess dir!.')
        print('Exit program...')
        sys.exit(1)

    engagement_map = pickle.load(open(engagement_map_loc, 'rb'))
    lookup_durations = np.array(engagement_map['duration'])

    # load pandas dataframe if exists
    re_dataframe_path = '../re_regressors/data/predicted_re_sparse_df.csv'
    if os.path.exists(re_dataframe_path):
        re_data_f = pd.read_csv(re_dataframe_path, sep='\t')
    else:
        print('Relative engagement dataframe not found!')
        sys.exit(1)

    wp_dataframe_path = '../wp_regressors/data/predicted_wp_sparse_df.csv'
    if os.path.exists(wp_dataframe_path):
        wp_data_f = pd.read_csv(wp_dataframe_path, sep='\t')
    else:
        print('Watch percentage dataframe not found!')
        sys.exit(1)

    # mae and r2 list
    mae_list = []
    r2_list = []
    name_list = ['Duration', 'Content', 'Topic', 'CTopic', 'CPS', 'All', 'CSP']
    for name in name_list:
        mae_list.append(mean_absolute_error(wp_data_f['True'], wp_data_f[name]))
        r2_list.append(r2_score(wp_data_f['True'], wp_data_f[name]))

        converted_wp = to_watch_percentage(engagement_map, re_data_f['Test duration'].tolist(), re_data_f[name].tolist(),
                                           lookup_keys=lookup_durations)
        mae_list.append(mean_absolute_error(wp_data_f['True'], converted_wp))
        r2_list.append(r2_score(wp_data_f['True'], converted_wp))

    for i in range(len(name_list)):
        print('\n>>> {2} MAE scores for wp and converted wp: {0} - {1}'.format(mae_list[2 * i], mae_list[2 * i + 1],
                                                                               name_list[i]))
        print('>>> {2} R2  scores for wp and converted wp: {0} - {1}'.format(r2_list[2 * i], r2_list[2 * i + 1],
                                                                             name_list[i]))

    plot_barchart(mae_list, r2_list)
