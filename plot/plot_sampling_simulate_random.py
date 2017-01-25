#!/usr/bin/python

# Usage example:
# python plot_sampling_simulate_random.py

import os
import json
import re
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = '../'

# twitter's snowflake parameters
twepoch = 1288834974657L
datacenter_id_bits = 5L
worker_id_bits = 5L
sequence_id_bits = 12L
max_datacenter_id = 1 << datacenter_id_bits
max_worker_id = 1 << worker_id_bits
max_sequence_id = 1 << sequence_id_bits
max_timestamp = 1 << (64L - datacenter_id_bits - worker_id_bits - sequence_id_bits)


def melt(snowflake_id, twepoch=twepoch):
    """inversely transform a snowflake id back to its parts."""
    sequence_id = snowflake_id & (max_sequence_id - 1)
    worker_id = (snowflake_id >> sequence_id_bits) & (max_worker_id - 1)
    datacenter_id = (snowflake_id >> sequence_id_bits >> worker_id_bits) & (max_datacenter_id - 1)
    timestamp_ms = snowflake_id >> sequence_id_bits >> worker_id_bits >> datacenter_id_bits
    timestamp_ms += twepoch

    return timestamp_ms, int(datacenter_id), int(worker_id), int(sequence_id)


def simulate_sampling():
    rate_path = os.path.join(BASE_DIR, 'data/rate_dict.json')

    # observed tweets in second
    observe_dict = defaultdict(int)
    with open(sample_path_25m, 'r') as sample_data:
        for line in sample_data:
            tweet = json.loads(line.rstrip())
            timestamp_s = tweet['timestamp_ms'][:-3]
            observe_dict[timestamp_s] += 1

    # sampling ratio in second
    sampling_ratio_dict = {}
    with open(rate_path, 'r') as rate_json:
        rate_dict = json.loads(rate_json.readline().rstrip())
        for t in sorted(observe_dict.keys()):
            if t in rate_dict:
                sampling_ratio_dict[t] = 1.0 * observe_dict[t] / (observe_dict[t] + rate_dict[t])
            # if not exist in rate dict, then retrieve complete set at that second
            else:
                sampling_ratio_dict[t] = 1.0

    # simulation procedure for 100 times
    for i in xrange(100):
        simulate_path = os.path.join(simulate_dir_25m, 'simulate_data_{0:03d}.json'.format(i))
        simulate_data = open(simulate_path, 'w+')
        with open(complete_path_25m, 'r') as complete_data:
            for line in complete_data:
                tweet = json.loads(line.rstrip())
                id = tweet['id']
                timestamp_s = str(melt(id)[0])[:-3]
                sampling_ratio = sampling_ratio_dict[timestamp_s]
                # generate float random uniformly in [0.0, 1.0)
                toss = random.random()
                if toss < sampling_ratio:
                    simulate_data.write(line)
        simulate_data.close()


def extract_vid_from_expanded_url(expanded_url):
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


def extract_vids(tweet):
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
        vid = extract_vid_from_expanded_url(expanded_url)
        if vid is not None:
            vids.add(vid)
    return vids


def plot_tweetcount(filepath, ax, label_text, color, style):
    tweetcount_dict = defaultdict(int)
    with open(filepath, 'r') as filedata:
        for line in filedata:
            tweet = json.loads(line.rstrip())
            vids = extract_vids(tweet)
            for vid in vids:
                tweetcount_dict[vid] += 1
    tc_stat_dict = defaultdict(int)
    tweetcounts = tweetcount_dict.values()
    for tc in tweetcounts:
        tc_stat_dict[tc] += 1
    x_axis = sorted(set(tweetcounts))
    y_axis = []
    residual = len(tweetcounts)
    for idx, x in enumerate(x_axis):
        if idx > 0:
            residual -= tc_stat_dict[x_axis[idx - 1]]
        y_axis.append(residual)
    ax.plot(x_axis, y_axis, style, c=color, ms=4, label=label_text, markeredgecolor=color)
    ax.set_xscale('symlog')
    ax.set_yscale('symlog')
    ax.set_xlim(xmin=0.8)
    ax.set_ylim(ymin=0.8)
    ax.set_xlabel('Number of tweets')
    ax.set_ylabel(r'Number of videos with $\geq$ x tweets')


def get_tweetcount(filepath):
    tweetcount_dict = defaultdict(int)
    with open(filepath, 'r') as filedata:
        for line in filedata:
            tweet = json.loads(line.rstrip())
            vids = extract_vids(tweet)
            for vid in vids:
                tweetcount_dict[vid] += 1
    return tweetcount_dict


def get_top_recall(path1, path2, threshold):
    complete_tweetcount = get_tweetcount(path1)
    sample_videos = set(get_tweetcount(path2).keys())
    top_complete_videos = set()
    for k, v in complete_tweetcount.items():
        if v > threshold-1:
            top_complete_videos.add(k)
    diff_videos = top_complete_videos.difference(sample_videos)
    print '{0} videos tweeted at least {1} times not seen in sample set'.format(len(diff_videos), threshold)
    print '{0} videos tweeted at least {1} times are in complete set'.format(len(top_complete_videos), threshold)
    print 'recall rate: {0:.2f}%'.format(100.0*len(diff_videos)/len(top_complete_videos))
    print '-------------------------------'


if __name__ == '__main__':
    sample_path_25m = os.path.join(BASE_DIR, 'data/sample_data.json')
    complete_path_25m = os.path.join(BASE_DIR, 'data/complete_data.json')
    simulate_path_25m_1 = os.path.join(BASE_DIR, 'data/simulate_data/simulate_data_048.json')
    simulate_path_25m_2 = os.path.join(BASE_DIR, 'data/simulate_data/simulate_data_088.json')
    simulate_dir_25m = os.path.join(BASE_DIR, 'data/simulate_data')

    # simulate_sampling()

    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig, ax1 = plt.subplots(1, 1)

    plot_tweetcount(simulate_path_25m_2, ax1, 'random sampling', 'r', 'o-')
    plot_tweetcount(complete_path_25m, ax1, 'firehose', 'b', '.-')
    plot_tweetcount(sample_path_25m, ax1, 'filter streaming', 'k', '|-')

    plt.legend(loc='best')
    plt.show()

    get_top_recall(complete_path_25m, sample_path_25m, 4)

    sample_path_10m = os.path.join(BASE_DIR, 'data/sample_data_10m.json')
    complete_path_10m = os.path.join(BASE_DIR, 'data/complete_data_10m.json')
    sample_path_5m = os.path.join(BASE_DIR, 'data/sample_data_5m.json')
    complete_path_5m = os.path.join(BASE_DIR, 'data/complete_data_5m.json')

    get_top_recall(complete_path_10m, sample_path_10m, 4)

    get_top_recall(complete_path_5m, sample_path_5m, 4)
