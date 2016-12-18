#!/usr/bin/python

# Usage example:
# python parse_dt_report.py --input='<input_file>' --output='<output_file>'

import os
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

BASE_DIR = '../'


if __name__ == '__main__':

    fig, ax1 = plt.subplots(1, 1)

    input_path1 = os.path.join(BASE_DIR, 'data/parsed-report-test.txt')
    input_path2 = os.path.join(BASE_DIR, 'data/parsed-nectar-test.txt')

    dt_dict = defaultdict(list)
    nectar_dict = defaultdict(list)

    with open(input_path1, 'r') as input_data1:
        for line in input_data1:
            if line.startswith('id'):
                id = line.rstrip().split(': ')[1]
            elif line.startswith('posted_time'):
                dt = line.rstrip().split(':  ', 1)[1]
                # time.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
                dt_obj = datetime.strptime(dt, '%m/%d/%Y %H:%M:%S')
                dt_dict[dt_obj].append(id)

    x_axis1 = sorted(dt_dict.keys())
    y_axis1 = [len(dt_dict[k]) for k in x_axis1]
    ax1.plot_date(x_axis1, y_axis1, 'o-', c='b', ms=3, label='discovertext')

    with open(input_path2, 'r') as input_data2:
        for line in input_data2:
            if line.startswith('id'):
                id = line.rstrip().split(': ')[1]
            elif line.startswith('posted_time'):
                dt = line.rstrip().split(': ', 1)[1]
                dt_obj = datetime.strptime(dt, '%a %b %d %H:%M:%S +0000 %Y')
                nectar_dict[dt_obj].append(id)

    x_axis2 = sorted(nectar_dict.keys())
    y_axis2 = [len(nectar_dict[k]) for k in x_axis2]
    ax1.plot_date(x_axis2, y_axis2, 'x-', c='r', ms=3, label='NeCTAR')

    for d in sorted(nectar_dict.keys()):
        discovert = set(dt_dict[d])
        nectar = set(nectar_dict[d])
        print 'for datetime {0}'.format(d)
        print 'tweets in discovertext: {0}'.format(len(discovert))
        print 'tweets in nectar: {0}'.format(len(nectar))
        tmp = discovert
        tmp = tmp.intersection(nectar)
        print 'tweets in both: {0}'.format(len(tmp))
        print '------------'

    plt.legend(loc='best')
    plt.show()
