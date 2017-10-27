#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Scripts to construct channel dataset.

Usage: python construct_channel_dataset.py input_doc output_doc
Example: python construct_channel_dataset.py ../../production_data/new_tweeted_dataset_norm ../../production_data/new_tweeted_channel_dataset
Time: ~18M
"""

from __future__ import division, print_function
import os, sys, time, datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    print('>>> Start to extract videos from one channel into one file...\n')
    start_time = time.time()

    input_loc = sys.argv[1]
    output_loc = sys.argv[2]
    if not os.path.exists(output_loc):
        os.mkdir(output_loc)

    # == == == == == == == == Part 2: Construct channel dataset == == == == == == == == #
    for subdir, _, files in os.walk(input_loc):
        for f in files:
            sub_f = os.path.basename(subdir)
            with open(os.path.join(subdir, f), 'r') as fin:
                fin.readline()
                for line in fin:
                    channel = line.rstrip().split('\t')[6]
                    if not os.path.exists(os.path.join(output_loc, sub_f, channel[:4])):
                        os.makedirs(os.path.join(output_loc, sub_f, channel[:4]))
                    with open(os.path.join(output_loc, sub_f, channel[:4], channel), 'a') as fout:
                        fout.write(line)
            print('>>> Finish converting file {0}!'.format(os.path.join(subdir, f)))

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])
