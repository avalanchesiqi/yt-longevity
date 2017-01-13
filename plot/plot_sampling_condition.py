#!/usr/bin/python

# Usage example:
# python plot_sampling_condition.py --sample='<sample_file>' --complete='<complete_file>'

import os
import argparse
import re
import json
from collections import defaultdict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = '../'
xFmt = mdates.DateFormatter('%H:%M:%S')


def _extract_vid_from_expanded_url(expanded_url):
    if 'watch?' in expanded_url and 'v=' in expanded_url:
        vid = expanded_url.split('v=')[1][:11]
    elif 'youtu.be' in expanded_url:
        vid = expanded_url.rsplit('/', 1)[-1][:11]
    else:
        return None
    # valid condition: contains only alphanumeric, dash or underline
    valid = re.match('^[\w-]+$', vid) is not None
    if valid and len(vid) == 11:
        return vid
    return None


def _extract_vids(tweet):
    urls = []
    if 'entities' in tweet.keys() and 'urls' in tweet['entities']:
        urls.extend(tweet['entities']['urls'])
    if 'retweeted_status' in tweet.keys():
        if 'entities' in tweet['retweeted_status'] and 'urls' in tweet['retweeted_status']['entities']:
            urls.extend(tweet['retweeted_status']['entities']['urls'])
        if 'extended_tweet' in tweet['retweeted_status']:
            if 'entities' in tweet['retweeted_status']['extended_tweet'] and 'urls' in \
                    tweet['retweeted_status']['extended_tweet']['entities']:
                urls.extend(tweet['retweeted_status']['extended_tweet']['entities']['urls'])
    if 'quoted_status' in tweet.keys():
        if 'entities' in tweet['quoted_status'] and 'urls' in tweet['quoted_status']['entities']:
            urls.extend(tweet['quoted_status']['entities']['urls'])
    expanded_urls = []
    for url in urls:
        if url['expanded_url'] is not None:
            expanded_urls.append(url['expanded_url'])

    vids = set()
    for expanded_url in expanded_urls:
        vid = _extract_vid_from_expanded_url(expanded_url)
        if vid is not None:
            vids.add(vid)
    return vids


def plot_query_vid(qvid):
    sample_dt_dict = defaultdict(int)
    with open(sample_path, 'r') as f:
        for line in f:
            tweet = json.loads(line.rstrip())
            dt_obj = datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
            vids = _extract_vids(tweet)
            for vid in vids:
                if vid == qvid:
                    sample_dt_dict[dt_obj] += 1

    complete_dt_dict = defaultdict(int)
    with open(complete_path, 'r') as f:
        for line in f:
            tweet = json.loads(line.rstrip())
            dt_obj = datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
            vids = _extract_vids(tweet)
            for vid in vids:
                if vid == qvid:
                    complete_dt_dict[dt_obj] += 1

    dt_x_axis1 = [datetime(2017, 1, 5, 23, 59, 40) + timedelta(seconds=i) for i in xrange(1447)]
    complete_y_axis1 = [complete_dt_dict[d] if d in complete_dt_dict else 0 for d in dt_x_axis1]
    sample_y_axis1 = [sample_dt_dict[d] if d in sample_dt_dict else 0 for d in dt_x_axis1]

    ax1.plot_date(dt_x_axis1, complete_y_axis1, '-', ms=1, label='complete'.rjust(8) + ' {0}: {1}'.format(qvid, sum(complete_y_axis1)))
    ax1.plot_date(dt_x_axis1, sample_y_axis1, '-', ms=1, label='sample'.rjust(9) + ' {0}: {1}'.format(qvid, sum(sample_y_axis1)))


if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sample', help='input file path of sample data, relative to base dir', required=True)
    parser.add_argument('-c', '--complete', help='output file path of complete data, relative to base dir',
                        required=True)
    args = parser.parse_args()

    sample_path = os.path.join(BASE_DIR, args.sample)
    complete_path = os.path.join(BASE_DIR, args.complete)

    fig, ax1 = plt.subplots(1, 1)

    plot_query_vid('3ronn0EFXtg')
    plot_query_vid('NcoubpSwlEk')
    plot_query_vid('g0o53b7L2cQ')

    ax1.set_ylabel('Number of tweets')
    ax1.set_title('Figure 1: complete and sample dataset in millisecond interval')
    ax1.xaxis.set_major_formatter(xFmt)
    ax1.set_xlabel('Jan\' 06 2017 UTC')
    ax1.legend(loc='best')

    plt.show()
