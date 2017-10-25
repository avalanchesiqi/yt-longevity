#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extract channel engagement from training dataset."""

import os, time, datetime


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    start_time = time.time()

    print('>>> Start to extract channel past engagement from training dataset...')

    if not os.path.exists('./data'):
        os.mkdir('./data')

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    data_loc = '../../production_data/tweeted_dataset_norm'
    train_loc = os.path.join(data_loc, 'train_data')

    output_data = open('./data/train_channel_watch_percentage.txt', 'w')
    for subdir, _, files in os.walk(train_loc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                fin.readline()
                for line in fin:
                    _, _, _, _, _, _, channel, _, _, _, wp30, _, _ = line.rstrip().split('\t', 12)
                    output_data.write('{0}\t{1}\n'.format(channel, wp30))

    output_data.close()

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])
