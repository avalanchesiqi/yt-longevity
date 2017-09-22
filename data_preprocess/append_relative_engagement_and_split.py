#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Scripts to append relative engagement to tweeted(quality) and split into train/test dataset.
Train: 2016-07-01 to 2016-08-21
Test: 2016-08-22 to 2016-08-31

Usage: python append_relative_engagement_and_split.py input_doc output_doc
Example: python append_relative_engagement_and_split.py ../../production_data/new_tweeted_dataset ../../production_data/new_tweeted_dataset_norm
Time: ~14M
"""

from __future__ import division, print_function
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import time
from datetime import timedelta
import cPickle as pickle

from utils.converter import to_relative_engagement


def extract_info(input_path, output_loc):
    """
    Append relative engagement map to dataset
    :param input_path: input file path
    :param output_loc: output root dir path
    :return:
    """
    f_train = open(os.path.join(output_loc, 'train', os.path.basename(input_path)), 'w')
    f_train.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\t{13}\t{14}\n'
                  .format('id', 'publish', 'duration', 'definition', 'category', 'detect_lang', 'channel', 'topics',
                          'view@30', 'watch@30', 'wp@30', 're@30', 'days', 'daily_view', 'daily_watch'))
    f_test = open(os.path.join(output_loc, 'test', os.path.basename(input_path)), 'w')
    f_test.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\t{13}\t{14}\n'
                 .format('id', 'publish', 'duration', 'definition', 'category', 'detect_lang', 'channel', 'topics',
                         'view@30', 'watch@30', 'wp@30', 're@30', 'days', 'daily_view', 'daily_watch'))

    with open(input_path, 'r') as fin:
        fin.readline()
        for line in fin:
            head, days, daily_view, daily_watch = line.rsplit('\t', 3)
            line_content = line.rstrip().split('\t')
            published_at = line_content[1]
            duration = int(line_content[2])
            wp30 = float(line_content[10])
            re30 = to_relative_engagement(lookup_table=engagement_map, duration=duration, wp_score=wp30, lookup_keys=duration_splits)

            if published_at < '2016-08-21':
                f_train.write('{0}\t{1}\t{2}\t{3}\t{4}'.format(head, re30, days, daily_view, daily_watch))
            else:
                f_test.write('{0}\t{1}\t{2}\t{3}\t{4}'.format(head, re30, days, daily_view, daily_watch))

    f_train.close()
    f_test.close()


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    print('>>> Start to append relative engagement and split to train/test dataset...')
    start_time = time.time()

    if not os.path.exists('engagement_map.p'):
        print('>>> No engagement map found! Run extract_engagement_map.py first!')
        sys.exit(1)

    # load engagement map
    engagement_map = pickle.load(open('engagement_map.p', 'r'))
    duration_splits = engagement_map['duration']

    input_loc = sys.argv[1]
    output_loc = sys.argv[2]
    if not os.path.exists(os.path.join(output_loc, 'train')):
        os.makedirs(os.path.join(output_loc, 'train'))
        os.makedirs(os.path.join(output_loc, 'test'))

    # == == == == == == == == Part 2: Construct dataset == == == == == == == == #
    for subdir, _, files in os.walk(input_loc):
        for f in files:
            extract_info(os.path.join(subdir, f), output_loc)
            print('>>> Finish converting file {0}!'.format(os.path.join(subdir, f)))

    # get running time
    print('\n>>> Total running time: {0}'.format(str(timedelta(seconds=time.time() - start_time)))[:-3])
