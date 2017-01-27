#!/usr/bin/python

# Usage example:
# python plot_cmp_discovertext_pstream.py

import os
import json
import re
import string
import numpy as np
import operator
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = '../'
xFmt = mdates.DateFormatter('%H:%M:%S')


# Determine whether a tweet matches "youtube" or "youtu" filter
def _match_filter(tweet):
    id = tweet['id_str']
    text = tweet['text'].lower()
    text = [word.strip(string.punctuation) for word in text.split()]
    if 'youtube' in text or 'youtu' in text:
        return id, True
    if 'entities' in tweet.keys():
        user_mentions = []
        if 'user_mentions' in tweet['entities']:
            user_mentions.extend([user_mention['name'].lower() for user_mention in tweet['entities']['user_mentions']])
            if 'youtube' in user_mentions or 'youtu' in user_mentions:
                return id, True
        hashtags = []
        if 'hashtags' in tweet['entities']:
            hashtags.extend([hashtag['text'].lower() for hashtag in tweet['entities']['hashtags']])
            if 'youtube' in hashtags or 'youtu' in hashtags:
                return id, True
        urls = []
        if 'urls' in tweet['entities']:
            forest = [re.findall(r"[\w']+", url['expanded_url']) for url in tweet['entities']['urls']]
            urls.extend([leaf for tree in forest for leaf in tree])
            if 'youtube' in urls or 'youtu' in urls:
                return id, True
    if 'retweeted_status' in tweet.keys():
        retweeted_status = tweet['retweeted_status']
        _, flag = _match_filter(retweeted_status)
        if flag:
            return id, True
        if 'extended_tweet' in tweet['retweeted_status']:
            extended_tweet = tweet['retweeted_status']['extended_tweet']
            _, flag = _match_filter(extended_tweet)
            if flag:
                return id, True
    if 'quoted_status' in tweet.keys():
        quoted_status = tweet['quoted_status']
        _, flag = _match_filter(quoted_status)
        if flag:
            return id, True
    return id, False


# Binarize miss tweets into match and mismatch set
def binarize_miss_tweets(filepath):
    match_tweets = set()
    mismatch_tweets = set()
    with open(filepath, 'r') as f:
        for line in f:
            tweet = json.loads(line.rstrip())
            id, flag = _match_filter(tweet)
            if flag:
                match_tweets.add(id)
            else:
                mismatch_tweets.add(id)
    return match_tweets, mismatch_tweets


if __name__ == '__main__':

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    discovertext_path = os.path.join(BASE_DIR, 'data/parsed-report-new.txt')
    nectar_path = os.path.join(BASE_DIR, 'data/parsed-nectar-new.txt')
    rate_path = os.path.join(BASE_DIR, 'data/rate_dict.json')
    miss_tweets_content_path = os.path.join(BASE_DIR, 'data/miss_tweets_content.txt')
    match_tweets, mismatch_tweets = binarize_miss_tweets(miss_tweets_content_path)
    miss_tweets_username_id_path = os.path.join(BASE_DIR, 'data/miss_tweets_username_id.txt')

    dt_dict = defaultdict(list)
    nectar_dict = defaultdict(list)
    id_username_dict = {}

    # discovertext report
    with open(discovertext_path, 'r') as dt_data:
        for line in dt_data:
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
    x_axis1 = x_axis1[20:-1]
    y_axis1 = [len(set(dt_dict[d])) for d in x_axis1]
    # ax1.plot_date(x_axis1, y_axis1, '-', c='b', ms=1, label='discovertext')

    # nectar report
    with open(nectar_path, 'r') as nectar_data:
        for line in nectar_data:
            if line.startswith('id'):
                id = line.rstrip().split(': ')[1]
            elif line.startswith('posted_time'):
                dt = line.rstrip().split(': ', 1)[1]
                dt_obj = datetime.strptime(dt, '%a %b %d %H:%M:%S +0000 %Y')
                nectar_dict[dt_obj].append(id)

    y_axis2 = [len(set(nectar_dict[d])) for d in x_axis1]
    # plot nectar (public stream) behavior
    ax1.plot_date(x_axis1, y_axis2, '-', c='g', ms=1, label='public streaming')

    # rate limit in seconds
    datetime_rate_dict = defaultdict(int)
    with open(rate_path, 'r') as rate_json:
        rate_dict = json.loads(rate_json.readline().rstrip())
        for t in sorted(rate_dict.keys()):
            datetime_rate_dict[datetime.utcfromtimestamp(int(t))] = rate_dict[t]

    y_axis3 = [len(set(nectar_dict[d]))+datetime_rate_dict[d] if d in datetime_rate_dict else len(set(nectar_dict[d])) for d in x_axis1]
    ax1.plot_date(x_axis1, y_axis3, '-', c='r', ms=1, label='public streaming + rate limit')

    # segment the difference of public stream and discovertext into match, mismatch, attrition
    match_dict = defaultdict(int)
    mismatch_dict = defaultdict(int)
    attrition_dict = defaultdict(int)
    miss_tweets_username_id = open(miss_tweets_username_id_path, 'w+')
    for d in x_axis1:
        dt_set = set(dt_dict[d])
        nectar_set = set(nectar_dict[d])
        print 'for datetime {0}'.format(d)
        print 'tweets in discovertext: {0}'.format(len(dt_set))
        print 'tweets in nectar: {0}'.format(len(nectar_set))
        print 'tweets in both: {0}'.format(len(dt_set.intersection(nectar_set)))
        miss_set = dt_set.difference(nectar_set)
        # print 'tweets in discovertext but not in nectar: {0}'.format(len(target))
        for id in miss_set:
            miss_tweets_username_id.write('{0} {1}\n'.format(id_username_dict[id], id))
            if id in match_tweets:
                match_dict[d] += 1
            elif id in mismatch_tweets:
                mismatch_dict[d] += 1
            else:
                attrition_dict[d] += 1
        print 'tweets missed from down sampling: {0}'.format(datetime_rate_dict[d])
        print 'tweets in nectar but not in discovertext: {0}'.format(len(nectar_set.difference(dt_set)))
        print 'total tweets in nectar and discovertext: {0}'.format(len(dt_set.union(nectar_set)))
        print '------------'
    miss_tweets_username_id.close()

    # plot discovertext - mismatch behavior
    y_axis4 = [len(set(dt_dict[d])) - mismatch_dict[d] for d in x_axis1]
    ax1.plot_date(x_axis1, y_axis4, '-', c='b', ms=1, label='discovertext - mismatch')

    # discovertext - mismatch behavior - attrition
    y_axis5 = [len(set(nectar_dict[d])) + match_dict[d] for d in x_axis1]
    # ax1.plot_date(x_axis1, y_axis5, '-', c='m', ms=1, label='public streaming + match')

    # consider all attrition tweets as match, upper bound curve
    upper_bound = [y_axis4[d] - y_axis3[d] for d in xrange(len(x_axis1))]

    # consider all attrition tweets as mismatch, lower bound curve
    lower_bound = [y_axis5[d] - y_axis3[d] for d in xrange(len(x_axis1))]

    # metric distortion rate
    benchmarks = [y_axis4, y_axis5]
    distortions = []
    for benchmark in benchmarks:
        ret = []
        for k in xrange(len(x_axis1)):
            ret.append(100.0 * abs(y_axis3[k] - benchmark[k]) / benchmark[k])
        distortions.append(np.mean(ret))

    ax2.plot((x_axis1[0], x_axis1[-1]), (0, 0), 'k-')
    ax2.plot_date(x_axis1, upper_bound, '-', c='m', ms=1, label='(discovertext - mismatch) - (public streaming + rate limit)')
    ax2.text(datetime(2017, 1, 6, 0, 0, 0), -150, r'mean: {0:2.2f}, std: {1:2.2f}, 1qr: {2:2.2f}, 3qr: {3:2.2f}, $\tau$: {4:2.2f}%'
             .format(np.mean(upper_bound), np.std(upper_bound), np.percentile(upper_bound, 25), np.percentile(upper_bound, 75), distortions[0]), fontsize=14)

    ax2.plot_date(x_axis1, lower_bound, '-', c='c', ms=1, label='(discovertext - mismatch - attrition) - (public streaming + rate limit)')
    ax2.text(datetime(2017, 1, 6, 0, 0, 0), -168, r'mean: {0:2.2f}, std: {1:2.2f}, 1qr: {2:2.2f}, 3qr: {3:2.2f}, $\tau$: {4:2.2f}%'
             .format(np.mean(lower_bound), np.std(lower_bound), np.percentile(lower_bound, 25), np.percentile(lower_bound, 75), distortions[1]), fontsize=14)

    ax2.text(datetime(2017, 1, 6, 0, 0, 0), -186, r'match tweets in miss tweets: {0:2.2f}%, mismatch tweets in miss tweets: {1:2.2f}%'
             .format(100.0*len(match_tweets)/(len(match_tweets)+len(mismatch_tweets)), 100.0*len(mismatch_tweets)/(len(match_tweets)+len(mismatch_tweets))), fontsize=14)

    ax1.set_ylabel('Number of tweets')
    ax1.set_title('Figure 1: discovertext and Public Streaming APIs behavior comparison')
    ax1.legend(loc='upper left')

    ax2.set_xlabel('Jan\' 06 2017 UTC')
    ax2.set_ylabel('Number of difference')
    ax2.set_title('Figure 2: difference between discovertext and Public Streaming APIs with rate limit and attrition')
    ax2.xaxis.set_major_formatter(xFmt)
    ax2.legend(loc='lower left')

    plt.show()
