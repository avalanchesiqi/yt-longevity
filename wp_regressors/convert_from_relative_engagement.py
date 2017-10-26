#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert watch percentage from relative engagement dataframe.

Example rows
   Unnamed: 0  Vid   True   Content     Topic    CTopic       CPS       All        CSP
0           0    1  0.481  0.607106  0.609333  0.665838  0.870050  0.876945   0.876945
1           1    1  0.295  0.507591  0.634284  0.514392  0.422758  0.488751   0.376236
"""

from __future__ import division, print_function
import os, sys, time, datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import pandas as pd
import cPickle as pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from utils.converter import to_watch_percentage


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    start_time = time.time()

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

    wp_dataframe_path = './data/predicted_wp_sparse_df.csv'
    if os.path.exists(wp_dataframe_path):
        wp_data_f = pd.read_csv(wp_dataframe_path, sep='\t')
    else:
        print('Watch percentage dataframe not found!')
        sys.exit(1)

    # mae and r2 list
    mae_list = []
    r2_list = []
    for name in ['Content', 'Topic', 'CTopic', 'CPS', 'All']:
        mae_list.append(mean_absolute_error(wp_data_f['True'], wp_data_f[name]))
        r2_list.append(r2_score(wp_data_f['True'], wp_data_f[name]))

        converted_wp = to_watch_percentage(engagement_map, re_data_f['Duration'], re_data_f[name], lookup_keys=lookup_durations)
        mae_list.append(mean_absolute_error(wp_data_f['True'], converted_wp))
        r2_list.append(r2_score(wp_data_f['True'], converted_wp))

    print('\n>>> MAE scores: ', mae_list)
    print('>>> R2 scores: ', r2_list)
