import os
from datetime import datetime
import json
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import dill
import bz2


def extract_vid(tweet):
    if 'entities' not in tweet.keys():
        raise Exception('No entities in tweet')
    urls = tweet['entities']['urls']
    num_urls = len(urls)
    if num_urls == 0:
        raise Exception('No urls in tweet')
    ret = []
    for i in xrange(num_urls):
        expanded_url = urls[i]['expanded_url']
        if 'watch?' in expanded_url and 'v=' in expanded_url:
            vid = expanded_url.split('v=')[1][:11]
        elif 'youtu.be' in expanded_url:
            vid = expanded_url.rsplit('/', 1)[-1][:11]
        else:
            continue
        # valid condition: contains only alphanumeric and dash
        valid = re.match('^[\w-]+$', vid) is not None
        if valid and len(vid) == 11:
            ret.append(vid)
    return ret


def get_tweet_rate(path):
    tweet_dict = defaultdict(int)
    with bz2.BZ2File(path, mode='r') as data:
        for line in data:
            if line.rstrip():
                res = json.loads(line.rstrip())
                if 'id' in res:
                    timestamp = int(res['timestamp_ms'][:-3])
                    dt = datetime.utcfromtimestamp(timestamp)
                    dt_minute = datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)
                    timestamp = (dt_minute - datetime(1970, 1, 1)).total_seconds() + 60
                    tweet_dict[timestamp] += 1
    return tweet_dict


if __name__ == '__main__':
    file_loc = '../../data/twitter-sampling'
    filename1 = '2016-12-01.bz2'
    filename2 = 'nectar.bz2'
    filepath1 = os.path.join(file_loc, filename1)
    filepath2 = os.path.join(file_loc, filename2)

    fig, (ax1, ax2) = plt.subplots(2, 1)

    tweet_dict1 = get_tweet_rate(filepath1)
    tweet_dict2 = get_tweet_rate(filepath2)

    start_timestamp = (datetime(2016, 11, 30, 13, 0, 0) - datetime(1970, 1, 1)).total_seconds()
    timestamp_axis = [start_timestamp + i*60 for i in xrange(24 * 60)]
    dml_axis = [tweet_dict1[ts] if ts in tweet_dict1 else 0 for ts in timestamp_axis]
    nectar_axis = [tweet_dict2[ts] if ts in tweet_dict2 else 0 for ts in timestamp_axis]

    diff_axis = [dml_axis[i] - nectar_axis[i] for i in xrange(len(timestamp_axis))]
    datetime_axis = [datetime.utcfromtimestamp(ts) for ts in timestamp_axis]

    ax1.plot_date(datetime_axis, dml_axis, '-', ms=2, color='r', label='dml machine')
    ax1.plot_date(datetime_axis, nectar_axis, '-', ms=2, color='g', label='NeCTAR machine')

    ax1.set_xlabel('Dec\' 01 2016 UTC')
    ax1.set_ylabel('Number of receive tweets')
    ax1.set_title('Figure 1: Minutely receive tweets number from dml and NeCTAR machine.')

    ax2.plot_date(datetime_axis, diff_axis, '-', ms=2, color='b', label='#tweets of (dml-NeCTAR)')
    ax2.set_xlabel('Dec\' 01 2016 UTC')
    ax2.set_ylabel('Number of difference betweet dml machine and NeCTAR machine')
    ax2.set_title('Figure 2: Difference betweet dml machine and NeCTAR machine.')

    plt.show()
