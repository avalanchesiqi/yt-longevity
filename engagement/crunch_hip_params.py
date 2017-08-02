#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import sys, os
import json
import bz2
import isodate
import numpy as np
import cPickle as pickle


# Extract HIP params and watch percentage

def time_decay(i, c):
    """
    Time decay part for series (tau + c).
    :param i: tau value
    :param c: c value
    :return: abbreviated presentation
    """
    return np.arange(1, i+1)[::-1]+c


def get_endo(params):
    _, theta, C, c, _, _ = params
    x_predict = np.zeros(10000)
    for i in xrange(10000):
        if i == 0:
            x_predict[0] = 1
        else:
            x_predict[i] = C*np.sum(x_predict[:i]*(time_decay(i, c)**(-1-theta)))
    return np.sum(x_predict)


def safe_div(wt, v, d):
    if np.sum(v) == 0 or d == 0:
        return 'NA'
    else:
        return np.sum(wt)*60/np.sum(v)/d


if __name__ == '__main__':
    # == == == == == == == == Part 1: Load ACTIVE dataset == == == == == == == == #
    # First time it gets loaded from the JSON format and writes essential fields into a pickle binary file.
    # check if the binary exists
    if not os.path.exists('./data/active-dataset.p'):
        print('>>> Converting ACTIVE dataset from JSON format to pickle... might take a while!')
        test_cases = {}
        with bz2.BZ2File('./data/active-dataset.json.bz2') as f:
            dataset = json.loads(f.readline())
            for video in dataset:
                if not video['duration'] == 'NA':
                    test_cases[video['YoutubeID']] = (video['numShare'], video['dailyViewcount'], video['watchTime'], video['duration'])
        pickle.dump(test_cases, open('./data/active-dataset.p', 'wb'))

    print('>>> Loading the ACTIVE dataset from pickle...')
    test_cases = pickle.load(open('./data/active-dataset.p', 'rb'))
    test_vids = test_cases.keys()
    print('number of videos in active dataset', len(test_vids))

    vid_params_dict = {}
    with open('./data/training_watch_reg_tune.log', 'r') as fin:
        for line in fin:
            vid, mu, theta, C, c, gamma, eta, _ = line.rstrip().split(None, 7)
            vid_params_dict[vid] = map(float, [mu, theta, C, c, gamma, eta])
    print('number of videos trained with HIP', len(vid_params_dict))

    output_file = open('./data/active_watch_reg_params.txt', 'w')
    output_file.write('vid\tduration\tmu\ttheta\tC\tc\tgamma\teta\texo\tendo\tviral\twp@30\twp@60\twp@90\twp@120\n')

    for tc_idx, vid in enumerate(test_vids):
        print('fitting and forecasting for video: {0}'.format(vid))
        _, dailyview, watchtime, duration_txt = test_cases[vid]
        duration = isodate.parse_duration(duration_txt).seconds

        # get watch percentage
        wp30 = safe_div(watchtime[:30], dailyview[:30], duration)
        wp60 = safe_div(watchtime[:60], dailyview[:60], duration)
        wp90 = safe_div(watchtime[:90], dailyview[:90], duration)
        wp120 = safe_div(watchtime[:120], dailyview[:120], duration)

        # get HIP params
        mu, theta, C, c, gamma, eta = vid_params_dict[vid]
        exo = mu
        endo = get_endo(vid_params_dict[vid])
        viral = exo*endo

        to_write = [vid, duration, mu, theta, C, c, gamma, eta, exo, endo, viral, wp30, wp60, wp90, wp120]
        output_file.write('{0}\n'.format('\t'.join(map(str, to_write))))

    output_file.close()
