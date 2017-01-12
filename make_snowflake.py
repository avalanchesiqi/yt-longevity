import datetime
from collections import defaultdict
import json
import os
import bz2
import string
import re
import matplotlib.pyplot as plt

# twitter's snowflake parameters
twepoch = 1288834974657L
datacenter_id_bits = 5L
worker_id_bits = 5L
sequence_id_bits = 12L
max_datacenter_id = 1 << datacenter_id_bits
max_worker_id = 1 << worker_id_bits
max_sequence_id = 1 << sequence_id_bits
max_timestamp = 1 << (64L - datacenter_id_bits - worker_id_bits - sequence_id_bits)


def make_snowflake(timestamp_ms, datacenter_id, worker_id, sequence_id, twepoch=twepoch):
    """generate a twitter-snowflake id, based on
    https://github.com/twitter/snowflake/blob/master/src/main/scala/com/twitter/service/snowflake/IdWorker.scala
    :param: timestamp_ms time since UNIX epoch in milliseconds"""

    sid = ((int(timestamp_ms) - twepoch) % max_timestamp) << datacenter_id_bits << worker_id_bits << sequence_id_bits
    sid += (datacenter_id % max_datacenter_id) << worker_id_bits << sequence_id_bits
    sid += (worker_id % max_worker_id) << sequence_id_bits
    sid += sequence_id % max_sequence_id

    return sid


def melt(snowflake_id, twepoch=twepoch):
    """inversely transform a snowflake id back to its parts."""
    sequence_id = snowflake_id & (max_sequence_id - 1)
    worker_id = (snowflake_id >> sequence_id_bits) & (max_worker_id - 1)
    datacenter_id = (snowflake_id >> sequence_id_bits >> worker_id_bits) & (max_datacenter_id - 1)
    timestamp_ms = snowflake_id >> sequence_id_bits >> worker_id_bits >> datacenter_id_bits
    timestamp_ms += twepoch

    return (timestamp_ms, int(datacenter_id), int(worker_id), int(sequence_id))


def utc_datetime(timestamp_ms):
    """convert millisecond timestamp to utc datetime object."""
    return datetime.datetime.utcfromtimestamp(timestamp_ms / 1000.)


# Determine whether a tweet matches "youtube" or "youtu" filter
def match_filter(tweet):
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
        _, flag = match_filter(retweeted_status)
        if flag:
            return id, True
        if 'extended_tweet' in tweet['retweeted_status']:
            extended_tweet = tweet['retweeted_status']['extended_tweet']
            _, flag = match_filter(extended_tweet)
            if flag:
                return id, True
    if 'quoted_status' in tweet.keys():
        quoted_status = tweet['quoted_status']
        _, flag = match_filter(quoted_status)
        if flag:
            return id, True
    return id, False


if __name__ == '__main__':
    fig, (ax1, ax2) = plt.subplots(2, 1)
    # fig, ax1 = plt.subplots(1, 1)
    tweet_stat = defaultdict(int)

    datacenter_worker_set = set()

    # with bz2.BZ2File('../data/twitter-sampling/2017-01-10.bz2', 'r') as filedata:
    #     for line in filedata:
    #         try:
    #             tweet = json.loads(line.rstrip())
    #             timestamp = int(tweet['timestamp_ms'])
    #             id = int(tweet['id_str'])
    #             # print datetime.datetime.fromtimestamp(melt(id)[0] / 1000.0), melt(id)[1:-1], melt(id)[-1]
    #             datacenter_worker_set.add(melt(id)[1:-1])
    #             tweet_stat[timestamp % 1000] += 1
    #         except:
    #             continue

    for subdir, _, files in os.walk('data/nectar_zips'):
        for f in sorted(files):
            if f.endswith('142860000.txt'):
                with open(os.path.join(subdir, f), 'r') as filedata:
                    for line in filedata:
                        try:
                            tweet = json.loads(line.rstrip())
                            timestamp_ms = int(tweet['timestamp_ms'])
                            # tweet_stat[timestamp_ms] += 1
                            ax1.scatter(timestamp_ms, 1, c='b', s=10, marker='o', lw = 0)
                            # print melt(id)[0], melt(id)[1:-1]
                        except:
                            continue

    with open('data/miss_tweets_content.txt') as f:
        for line in f:
            tweet = json.loads(line.rstrip())
            id, flag = match_filter(tweet)
            if flag:
                tweet_stat[melt(int(id))[0]%1000] += 1
                ax1.scatter(melt(int(id))[0], 1, c='r', s=30, marker='s', lw = 0)
                # print melt(int(id))[0], melt(int(id))[1:-1]
                # if 656 < melt(int(id))[0]%1000 < 667:
                #     print "caution!!!", id, melt(int(id))[0]

    with open('data/rate.txt') as f:
        for line in f:
            rate = json.loads(line.rstrip())
            timestamp_ms = rate['limit']['timestamp_ms']
            ax1.scatter(timestamp_ms, 1, c='g', s=30, marker='D', lw = 0)

    ax1.plot((1483661032657L, 1483661032657L), (0.999, 1.001), 'm')
    ax1.plot((1483661033657L, 1483661033657L), (0.999, 1.001), 'm')
    ax1.plot((1483661034657L, 1483661034657L), (0.999, 1.001), 'm')
    ax1.plot((1483661035657L, 1483661035657L), (0.999, 1.001), 'm')
    ax1.plot((1483661036657L, 1483661036657L), (0.999, 1.001), 'm')
    ax1.plot((1483661037657L, 1483661037657L), (0.999, 1.001), 'm')
    ax1.plot((1483661038657L, 1483661038657L), (0.999, 1.001), 'm')
    ax1.plot((1483661039657L, 1483661039657L), (0.999, 1.001), 'm')
    ax1.plot((1483661040657L, 1483661040657L), (0.999, 1.001), 'm')
    ax1.plot((1483661041657L, 1483661041657L), (0.999, 1.001), 'm')
    ax1.plot((1483661042657L, 1483661042657L), (0.999, 1.001), 'm')
    ax1.plot((1483661043657L, 1483661043657L), (0.999, 1.001), 'm')
    ax1.plot((1483661044657L, 1483661044657L), (0.999, 1.001), 'm')
    ax1.plot((1483661045657L, 1483661045657L), (0.999, 1.001), 'm')
    ax1.plot((1483661046657L, 1483661046657L), (0.999, 1.001), 'm')
    ax1.plot((1483661047657L, 1483661047657L), (0.999, 1.001), 'm')
    ax1.plot((1483661048657L, 1483661048657L), (0.999, 1.001), 'm')
    ax1.plot((1483661049657L, 1483661049657L), (0.999, 1.001), 'm')
    ax1.plot((1483661050657L, 1483661050657L), (0.999, 1.001), 'm')
    ax1.plot((1483661051657L, 1483661051657L), (0.999, 1.001), 'm')
    ax1.plot((1483661052657L, 1483661052657L), (0.999, 1.001), 'm')

    ax1.plot((1483661032666L, 1483661032666L), (0.999, 1.001), 'y')
    ax1.plot((1483661033666L, 1483661033666L), (0.999, 1.001), 'y')
    ax1.plot((1483661034666L, 1483661034666L), (0.999, 1.001), 'y')
    ax1.plot((1483661035666L, 1483661035666L), (0.999, 1.001), 'y')
    ax1.plot((1483661036666L, 1483661036666L), (0.999, 1.001), 'y')
    ax1.plot((1483661037666L, 1483661037666L), (0.999, 1.001), 'y')
    ax1.plot((1483661038666L, 1483661038666L), (0.999, 1.001), 'y')
    ax1.plot((1483661039666L, 1483661039666L), (0.999, 1.001), 'y')
    ax1.plot((1483661040666L, 1483661040666L), (0.999, 1.001), 'y')
    ax1.plot((1483661041666L, 1483661041666L), (0.999, 1.001), 'y')
    ax1.plot((1483661042666L, 1483661042666L), (0.999, 1.001), 'y')
    ax1.plot((1483661043666L, 1483661043666L), (0.999, 1.001), 'y')
    ax1.plot((1483661044666L, 1483661044666L), (0.999, 1.001), 'y')
    ax1.plot((1483661045666L, 1483661045666L), (0.999, 1.001), 'y')
    ax1.plot((1483661046666L, 1483661046666L), (0.999, 1.001), 'y')
    ax1.plot((1483661047666L, 1483661047666L), (0.999, 1.001), 'y')
    ax1.plot((1483661048666L, 1483661048666L), (0.999, 1.001), 'y')
    ax1.plot((1483661049666L, 1483661049666L), (0.999, 1.001), 'y')
    ax1.plot((1483661050666L, 1483661050666L), (0.999, 1.001), 'y')
    ax1.plot((1483661051666L, 1483661051666L), (0.999, 1.001), 'y')
    ax1.plot((1483661052666L, 1483661052666L), (0.999, 1.001), 'y')

    # print sorted(list(datacenter_worker_set))
    # print len(datacenter_worker_set)

    for d in tweet_stat.keys():
        # if d < 657:
        #     ax2.scatter(d+1000, tweet_stat[d], s=5, c='r')
        ax2.scatter(d, tweet_stat[d], s=5, c='b')

    ax1.set_xlim(xmin=1483661031657L)
    ax1.set_xlim(xmax=1483661053666L)
    ax1.set_xlabel('Timestamp in millisecond')
    ax1.set_ylabel('Number of tweets')
    ax1.set_title('Figure 1: number of tweets in 1000 millisecond buckets, 20s period')

    ax2.set_xlabel('Timestamp in millisecond')
    ax2.set_ylabel('Number of tweets')
    ax2.set_title('Figure 2: number of miss tweets in 1000 millisecond buckets, discovertext period')

    plt.show()
