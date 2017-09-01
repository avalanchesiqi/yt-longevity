#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot regressor prediction results for relative engagement."""

from __future__ import division, print_function
import sys, os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def plot_barchart(mae_list, r2_list):
    # generate barchart plot
    fig = plt.figure(figsize=(8, 6))
    width = 0.35
    ind = np.arange(6)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    cornflower_blue = '#6495ed'
    tomato = '#ff6347'

    ax1.bar(ind, mae_list, width, edgecolor='k', color=cornflower_blue)
    ax2.bar(ind, r2_list, width, edgecolor='k', color=tomato)

    ax1.set_ylim([0, 0.3])
    ax2.set_ylim([0, 1])
    ax1.set_yticks([0.1, 0.2, 0.3])
    ax2.set_yticks([0.2, 0.6, 1.0])
    ax1.set_ylabel(r'$MAE$', fontsize=20)
    ax2.set_ylabel(r'$R^2$', fontsize=20)

    for label in ax1.get_xticklabels()[:]:
        label.set_visible(False)
    ax2.set_xticklabels(('', 'C', 'T', 'C+T', 'CPS', 'ALL', 'CSP'))

    for tick, label in zip(ind, ax1.get_xticklabels()):
        ax1.text(ind[tick], mae_list[tick] + 0.01, [str(np.round(x, 4)) for x in mae_list][tick],
                 horizontalalignment='center', color='k', fontsize=16)
    for tick, label in zip(ind, ax2.get_xticklabels()):
        ax2.text(ind[tick], r2_list[tick] + 0.01, [str(np.round(x, 4)) for x in r2_list][tick],
                 horizontalalignment='center', color='k', fontsize=16)

    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.tight_layout(rect=[0.01, 0, 1, 1])
    plt.show()


if __name__ == '__main__':
    # load pandas dataframe if exists
    dataframe_path = './data/predicted_sparse_df.csv'
    if os.path.exists(dataframe_path):
        data_f = pd.read_csv(dataframe_path, sep='\t')
    else:
        print('Data frame not generated yet, go back to construct_pandas_frame.py!')
        sys.exit(1)

    m, n = data_f.shape
    print('>>> Final dataframe size {0} instances with {1} features'.format(m, n))
    print('>>> Header of final dataframe')
    print(data_f.head())

    # mae and r2 list
    mae_list = []
    r2_list = []
    for name in ['Content', 'Topic', 'CTopic', 'CPS', 'All', 'CSP']:
        mae_list.append(mean_absolute_error(data_f['True'], data_f[name]))
        r2_list.append(r2_score(data_f['True'], data_f[name]))
    print('\n>>> MAE scores: ', mae_list)
    print('>>> R2 scores: ', r2_list)

    plot_barchart(mae_list, r2_list)
