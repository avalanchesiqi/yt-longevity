import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from sklearn import linear_model

file_loc = '../../data/new_tweets'

if __name__ == '__main__':
    trans_lst = []
    for subdir, _, files in os.walk(file_loc):
        for f in sorted(files):
            filepath = os.path.join(subdir, f)
            with open(filepath) as filedata:
                tweet_text = filedata.readline().rstrip()
                while tweet_text:
                    try:
                        tweet_json = json.loads(tweet_text)
                    except:
                        print 'json loads faile', tweet_text
                    try:
                        if 'id' in tweet_json:
                            server_ts = int(tweet_json['timestamp_ms'])
                        elif 'limit' in tweet_json:
                            server_ts = int(tweet_json['limit']['timestamp_ms'])
                    except:
                        print 'no timestamp'
                        print tweet_text
                    client_ts_text = filedata.readline().rstrip()
                    client_ts = int(client_ts_text)
                    duration = client_ts - server_ts
                    trans_lst.append(duration)
                    tweet_text = filedata.readline().rstrip()

    print max(trans_lst)
    print min(trans_lst)
    print np.mean(trans_lst)
    print np.median(trans_lst)

    fig, ax = plt.subplots(1, 1)
    ax.hist(trans_lst, 1000, normed=0, facecolor='green', alpha=0.75)
    ax.set_xlabel('Delay time in milliseconds')
    ax.set_ylabel('Number of volume')
    plt.show()