import os
from datetime import datetime, timedelta
import json
import re
from collections import defaultdict
import matplotlib.pyplot as plt
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
    rate_dict = defaultdict(int)
    prev_ts = ''
    prev_rates = []
    curr_rates = []
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
                else:
                    timestamp = int(round(1.0*int(res['limit']['timestamp_ms'])/1000))
                    rate = int(res['limit']['track'])
                    dt = datetime.utcfromtimestamp(timestamp)
                    dt_minute = datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)
                    timestamp_minute = (dt_minute - datetime(1970, 1, 1)).total_seconds() + 60
                    if not timestamp == prev_ts:
                        if len(curr_rates) == 0:
                            curr_rates.append(rate)
                        elif len(prev_rates) == 0:
                            prev_rates = curr_rates
                            curr_rates = []
                        else:
                            prev_rates = sorted(prev_rates, reverse=True)
                            curr_rates = sorted(curr_rates, reverse=True)
                            # each element in curr_rates should be larger than respective prev_rates element
                            for i in xrange(4):
                                if i >= len(curr_rates) or i >= len(prev_rates):
                                    break
                                if curr_rates[i] <= prev_rates[i]:
                                    curr_rates.insert(i, prev_rates[i])
                            # fulfill prev_rates
                            m = len(prev_rates)
                            n = len(curr_rates)
                            if m < n:
                                prev_rates.extend(curr_rates[m-n:])
                            elif m > n:
                                curr_rates.extend(prev_rates[n-m:])
                            diff = sum(curr_rates) - sum(prev_rates)
                            if diff < 0:
                                rate_dict[timestamp_minute] += 0
                                prev_rates = [0, 0, 0, 0]
                            else:
                                rate_dict[timestamp_minute] += diff
                                prev_rates = curr_rates
                            curr_rates = []
                    else:
                        curr_rates.append(rate)
                    prev_ts = timestamp
    return tweet_dict, rate_dict


if __name__ == '__main__':
    file_loc = '../../data/twitter-sampling'
    date = '2016-12-02'
    filename1 = '{0}.bz2'.format(date)
    filename2 = 'nectar_{0}.bz2'.format(date)
    # filename1 = 'dml_sample_tweets.bz2'
    # filename2 = 'nectar_sample_tweets.bz2'
    filepath1 = os.path.join(file_loc, filename1)
    filepath2 = os.path.join(file_loc, filename2)

    fig, (ax1, ax2) = plt.subplots(2, 1)

    tweet_dict1, rate_dict1 = get_tweet_rate(filepath1)
    tweet_dict2, rate_dict2 = get_tweet_rate(filepath2)

    start_timestamp = (datetime(*map(int, date.split('-'))) - timedelta(hours=11) - datetime(1970, 1, 1)).total_seconds()
    timestamp_axis = [start_timestamp + i*60 for i in xrange(24 * 60)]
    dml_axis = [tweet_dict1[ts] if ts in tweet_dict1 else 0 for ts in timestamp_axis]
    dml_rate_axis = [rate_dict1[ts] if ts in rate_dict1 else 0 for ts in timestamp_axis]
    dml_reconstruct_axis = [dml_axis[i]+dml_rate_axis[i] for i in xrange(len(timestamp_axis))]
    nectar_axis = [tweet_dict2[ts] if ts in tweet_dict2 else 0 for ts in timestamp_axis]
    nectar_rate_axis = [rate_dict2[ts] if ts in rate_dict2 else 0 for ts in timestamp_axis]
    nectar_reconstruct_axis = [nectar_axis[i]+nectar_rate_axis[i] for i in xrange(len(timestamp_axis))]

    diff_axis = [dml_axis[i] - nectar_axis[i] for i in xrange(len(timestamp_axis))]
    diff_reconstruct_axis = [dml_reconstruct_axis[i] - nectar_reconstruct_axis[i] for i in xrange(len(timestamp_axis))]
    datetime_axis = [datetime.utcfromtimestamp(ts) for ts in timestamp_axis]

    ax1.plot_date(datetime_axis, dml_axis, '-', ms=2, color='r', label='dml machine')
    ax1.plot_date(datetime_axis, dml_reconstruct_axis, '-', ms=2, color='b', label='dml machine reconstruct')
    ax1.plot_date(datetime_axis, nectar_axis, '-', ms=2, color='g', label='NeCTAR machine')
    ax1.plot_date(datetime_axis, nectar_reconstruct_axis, '-', ms=2, color='m', label='NeCTAR machine reconstruct')

    ax1.set_xlabel('Dec\' 02 2016 UTC')
    ax1.set_ylabel('Number of receive tweets')
    ax1.set_title('Figure 1: Minutely receive/reconstruct tweets number from dml and NeCTAR machine.')

    ax2.plot_date(datetime_axis, diff_axis, '-', ms=2, color='b', label='#tweets of (dml-NeCTAR)')
    ax2.plot_date(datetime_axis, diff_reconstruct_axis, '-', ms=2, color='m', label='#tweets of (dml-NeCTAR) reconstruct')
    ax2.set_xlabel('Dec\' 02 2016 UTC')
    ax2.set_ylabel('Number of difference betweet dml machine and NeCTAR machine')
    ax2.set_title('Figure 2: Difference betweet dml machine and NeCTAR machine.')

    ax1.legend(loc='upper right')
    ax2.legend(loc='lower right')

    plt.show()
