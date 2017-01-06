# Import the necessary package to process data in JSON format
try:
    import json
except ImportError:
    import simplejson as json

import os
import time
from collections import defaultdict
# Import the necessary methods from "twitter" library
from twitter import Twitter, OAuth, TwitterHTTPError

# Use yt.trend token set
ACCESS_TOKEN = '2478452719-iQcJt7U64cf7DSj4N7muOxC2B0UXNoHRuYgE25f'
ACCESS_SECRET = 'pcZ4U1LWZyROOuCbOYud8DMMXzhiAsWQyDInNeyjnA2Zt'
CONSUMER_KEY = '9vth8YgBzj6ctUXVm7eFk6t8J'
CONSUMER_SECRET = '6ZqBdSAD4hdd6hJGZTmbyicVcmB3LtvCiehLdBJB5Mp15A0egk'

BASE_DIR = '../..'


output = open(os.path.join(BASE_DIR, 'data/miss_tweets_json.txt'), 'a+')

def search_tweet(username, tweet_id):
    cnt = 0
    try:
        iterator = twitter_search.search.tweets(q='from:{0} since:2017-01-05 until:2017-01-07'.format(username), result_type='recent', count=100)
    except TwitterHTTPError as e:
        if 'Rate limit exceeded' in e.message:
            time.sleep(960)
        iterator = twitter_search.search.tweets(q='from:{0} since:2017-01-05 until:2017-01-07'.format(username), result_type='recent', count=100)
    flag = False

    if 'statuses' in iterator:
        for tweet in iterator['statuses']:
            cnt += 1
            if tweet['id_str'] == tweet_id:
                output.write(json.dumps(tweet))
                output.write('\n')
                # print json.dumps(tweet)
                flag = True
                break

    while 'next_results' in iterator['search_metadata'] and not flag:
        next_results = iterator['search_metadata']['next_results']
        max_id_ = next_results.split('max_id=')[1][:18]
        try:
            iterator = twitter_search.search.tweets(q='from:{0} since:2017-01-05 until:2017-01-07'.format(username), result_type='recent', count=100, max_id=max_id_)
        except TwitterHTTPError as e:
            if 'Rate limit exceeded' in e.message:
                time.sleep(960)
            iterator = twitter_search.search.tweets(q='from:{0} since:2017-01-05 until:2017-01-07'.format(username), result_type='recent', count=100, max_id=max_id_)
        if 'statuses' in iterator:
            cnt += 1
            for tweet in iterator['statuses']:
                if tweet['id_str'] == tweet_id:
                    output.write(json.dumps(tweet))
                    output.write('\n')
                    flag = True
                    break
    if flag:
        print 'find out-of-sample tweet {0} {1}'.format(username, tweet_id)
    else:
        print 'CAUTION: not find tweet {0} {1}'.format(username, tweet_id)
    print 'search from {0} tweets'.format(cnt)
    print '--------------------'


if __name__ == '__main__':
    oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

    # Initiate the connection to Twitter Streaming API
    # twitter_stream = TwitterStream(auth=oauth)

    # Initiate the connection to Twitter Search API
    twitter_search = Twitter(auth=oauth)

    # Search tweets by using following query
    to_search = defaultdict(list)
    with open(os.path.join(BASE_DIR, 'data/miss_tweets_new.txt'), 'r') as miss_tweets:
        for line in miss_tweets:
            username, tweet_id = line.rstrip().split()
            to_search[username].append(tweet_id)

    for username in to_search:
        for tweet_id in to_search[username]:
            search_tweet(username, tweet_id)
