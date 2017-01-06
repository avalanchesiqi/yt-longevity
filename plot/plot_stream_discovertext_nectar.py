#!/usr/bin/python

# Usage example:
# python parse_dt_report.py --input='<input_file>' --output='<output_file>'

import os
import json
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = '../'
xFmt = mdates.DateFormatter('%H:%M:%S')

if __name__ == '__main__':

    fig, ax1 = plt.subplots(1, 1)

    input_path1 = os.path.join(BASE_DIR, 'data/parsed-report-new.txt')
    input_path2 = os.path.join(BASE_DIR, 'data/parsed-nectar-new.txt')

    dt_dict = defaultdict(list)
    nectar_dict = defaultdict(list)
    id_username_dict = {}

    with open(input_path1, 'r') as input_data1:
        for line in input_data1:
            if line.startswith('username'):
                username = line.rstrip().split(': ')[1]
            elif line.startswith('id'):
                id = line.rstrip().split(': ')[1]
                id_username_dict[id] = username
            elif line.startswith('posted_time'):
                dt = line.rstrip().split(': ', 1)[1]
                # time.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
                dt_obj = datetime.strptime(dt, '%m/%d/%Y %H:%M:%S')
                dt_dict[dt_obj].append(id)

    x_axis1 = sorted(dt_dict.keys())
    y_axis1 = [len(dt_dict[k]) for k in x_axis1]
    ax1.plot_date(x_axis1, y_axis1, 'o-', c='b', ms=4, label='discovertext')

    with open(input_path2, 'r') as input_data2:
        for line in input_data2:
            if line.startswith('id'):
                id = line.rstrip().split(': ')[1]
            elif line.startswith('posted_time'):
                dt = line.rstrip().split(': ', 1)[1]
                dt_obj = datetime.strptime(dt, '%a %b %d %H:%M:%S +0000 %Y')
                nectar_dict[dt_obj].append(id)

    y_axis2 = [len(nectar_dict[k]) for k in x_axis1]
    ax1.plot_date(x_axis1, y_axis2, 'o-', c='r', ms=4, label='Public Streaming')

    dt_rate_dict = defaultdict(int)
    with open(os.path.join(BASE_DIR, 'data/rate_dict.json'), 'r') as rate_json:
        rate_dict = json.loads(rate_json.readline().rstrip())
        for t in sorted(rate_dict.keys()):
            dt_rate_dict[datetime.utcfromtimestamp(int(t))] = rate_dict[t]

    y_axis3 = [len(nectar_dict[k])+dt_rate_dict[k] if k in dt_rate_dict else len(nectar_dict[k]) for k in x_axis1]
    ax1.plot_date(x_axis1, y_axis3, 'o-', c='g', ms=4, label='Public Streaming + rate limit')

    miss_tweets_path = os.path.join(BASE_DIR, 'data/miss_tweets_new.txt')
    # remove output file if exists
    try:
        os.remove(miss_tweets_path)
    except OSError:
        pass

    miss_tweets = open(miss_tweets_path, 'a+')
    for d in sorted(dt_dict.keys()):
        discovert = set(dt_dict[d])
        nectar = set(nectar_dict[d])
        print 'for datetime {0}'.format(d)
        print 'tweets in discovertext: {0}'.format(len(discovert))
        print 'tweets in nectar: {0}'.format(len(nectar))
        tmp = discovert
        tmp = tmp.intersection(nectar)
        print 'tweets in both: {0}'.format(len(tmp))
        tmp = discovert
        tmp = tmp.difference(nectar)
        print 'tweets in discovertext but not in nectar: {0}'.format(len(tmp))
        for id in tmp:
            miss_tweets.write('{0} {1}\n'.format(id_username_dict[id], id))
        print 'tweets missed from down sampling: {0}'.format(dt_rate_dict[d])
        tmp = nectar
        tmp = tmp.difference(discovert)
        print 'tweets in nectar but not in discovertext: {0}'.format(len(tmp))
        tmp = nectar
        tmp = tmp.union(discovert)
        print 'total tweets in nectar and discovertext: {0}'.format(len(tmp))
        print '------------'

    miss_tweets.close()

    ax1.set_xlabel('Jan  \' 06 2017 UTC')
    ax1.set_ylabel('Number of tweets')
    ax1.set_title('Figure 1: discovertext and Public Streaming APIs behavior comparison')

    ax1.xaxis.set_major_formatter(xFmt)
    plt.legend(loc='lower right')
    plt.show()
