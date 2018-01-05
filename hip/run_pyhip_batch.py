#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import sys, os, bz2, json, time
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from datetime import timedelta
import cPickle as pickle

from pyhip import HIP
from utils.helper import strify

if __name__ == '__main__':
    # == == == == == == == == Part 1: Load ACTIVE dataset == == == == == == == == #
    # First time it gets loaded from the JSON format and writes essential fields into a pickle binary file.
    # check if the binary exists
    if not os.path.exists('./data/active-dataset.p'):
        print('>>> Converting ACTIVE dataset from JSON format to pickle... might take a while!')
        active_videos = {}
        with bz2.BZ2File('./data/active-dataset.json.bz2') as f:
            dataset = json.loads(f.readline())
            for video in dataset:
                active_videos[video['YoutubeID']] = (video['numShare'], video['dailyViewcount'], video['watchTime'])
        pickle.dump(active_videos, open('./data/active-dataset.p', 'wb'))

    print('>>> Loading the ACTIVE dataset from pickle...')
    active_videos = pickle.load(open('./data/active-dataset.p', 'rb'))

    # == == == == == == == == Part 2: Fit model and forecast future volume == == == == == == == == #
    num_train = 90
    num_test = 30
    num_initialization = 25
    age = num_train + num_test
    fout = open('./data/hip_view2.csv', 'w')
    fout.write('YoutubeID\t{0}\n'.format('\t'.join(['Day{0}'.format(i) for i in range(num_train+1, age+1)])))

    for vid in active_videos.keys():
        daily_share, daily_view, daily_watch = active_videos[vid]
        daily_attention = daily_view
        if len(daily_share) >= 120:
            start_time = time.time()
            hip_model = HIP()
            hip_model.initial(daily_share, daily_attention, num_train, num_test, num_initialization)
            hip_model.fit_with_bfgs()
            predicted_attention = hip_model.predict(hip_model.get_parameters_abbr(), daily_share[:age])[-num_test:]
            fout.write('{0}\t{1}\n'.format(vid, strify(predicted_attention, delimiter='\t')))
            print('>>> {0} fitting time: {1}'.format(vid, str(timedelta(seconds=time.time() - start_time)))[:-3])

    fout.close()
