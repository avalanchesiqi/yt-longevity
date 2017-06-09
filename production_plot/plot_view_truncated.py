#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import sys
import os
import json
import numpy as np
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    age = 182
    view_percent_matrix = None
    cnt = 0

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    data_loc = '../../data/production_data/'

    record_list = []
    for i in xrange(180):
        data_path = os.path.join(data_loc, '{0}.txt'.format(i))
        tmp = []
        with open(data_path, 'r') as fin:
            for line in fin:
                tmp.append(float(line.rstrip()))
        record_list.append(np.median(tmp))
        tmp = None
        print 'data {0} loaded!'.format(i)
    print 'ALL data loaded!'

    fig, ax1 = plt.subplots(1, 1)
    print sum(record_list)
    ax1.plot(np.arange(1, 181), record_list)
    ax1.set_xlabel('Age')
    ax1.set_ylabel('View Percentage')
    # ax1.set_yscale('log')

    plt.show()
