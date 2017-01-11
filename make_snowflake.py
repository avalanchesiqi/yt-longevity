import datetime
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

def local_datetime(timestamp_ms):
    """convert millisecond timestamp to local datetime object."""
    return datetime.datetime.fromtimestamp(timestamp_ms / 1000.)


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


if __name__ == '__main__':
    import time
    # t0 = int(time.time() * 1000)
    # print local_datetime(t0)
    # assert(melt(make_snowflake(t0, 0, 0, 0))[0] == t0)

    from collections import defaultdict

    stat = defaultdict(int)

    datacenter_worker_set = set()

    # with bz2.BZ2File('../data/twitter-sampling/nectar_2016-12-14.bz2', 'r') as filedata:
    #     for line in filedata:
    #         try:
    #             tweet = json.loads(line.rstrip())
    #             timestamp = int(tweet['timestamp_ms'])
    #             id = int(tweet['id_str'])
    #             print datetime.datetime.fromtimestamp(melt(id)[0] / 1000.0), melt(id)[1:-1], melt(id)[-1]
    #             datacenter_worker_set.add(melt(id)[1:-1])
    #             stat[timestamp % 1000] += 1
    #         except:
    #             continue

    for subdir, _, files in os.walk('data/nectar_zips'):
        for f in sorted(files):
            if f.endswith('000.txt'):
                with open(os.path.join(subdir, f), 'r') as filedata:
                    for line in filedata:
                        try:
                            tweet = json.loads(line.rstrip())
                            timestamp = int(tweet['timestamp_ms'])
                            id = int(tweet['id_str'])
                            print datetime.datetime.fromtimestamp(melt(id)[0]/1000.0), melt(id)[1:-1], melt(id)[-1]
                            datacenter_worker_set.add(melt(id)[1:-1])
                            stat[timestamp%1000] += 1
                        except:
                            continue

    with open('data/miss_tweets_content.txt') as f:
        for line in f:
            tweet = json.loads(line.rstrip())
            id, flag = _match_filter(tweet)
            if flag:
                stat[melt(int(id))[0]%1000] += 1

    print sorted(list(datacenter_worker_set))
    print len(datacenter_worker_set)

    fig, ax1 = plt.subplots(1, 1)
    for d in sorted(stat.keys()):
        if d > 656:
            ax1.scatter(d-1000, stat[d], s=5)
        else:
            ax1.scatter(d, stat[d], s=5)

    for i in xrange(657, 667):
        print stat[i]

    ax1.set_xlabel('Timestamp in millisecond')
    ax1.set_ylabel('Number of tweets')
    ax1.set_title('Figure 1: number of tweets allocated in 1000 millisecond buckets')
    plt.show()
