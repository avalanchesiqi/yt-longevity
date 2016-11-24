import os
from datetime import datetime
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdate


if __name__ == '__main__':
    file_loc = '../../data/twitter-sampling'
    filename = '2016-07-14_active.txt'
    filepath = os.path.join(file_loc, filename)

    prev_rates = [356878, 356932, 357029, 358653]
    prev_ts = 0
    cur_rates = []
    response_dict = defaultdict(int)
    rate_dict = defaultdict(int)
    with open(filepath, 'r') as filedata:
        for line in filedata:
            res = json.loads(line.rstrip())
            if 'id' in res:
                timestamp = int(res['timestamp_ms'][:-3])
                response_dict[timestamp] += 1
            else:
                timestamp = int(round(1.0 * int(res['limit']['timestamp_ms']) / 1000))
                rate = int(res['limit']['track'])
                if timestamp == prev_ts or prev_ts == 0:
                    cur_rates.append(rate)
                else:
                    if len(cur_rates) < len(prev_rates):
                        cur_rates = sorted(cur_rates, reverse=True)
                        prev_rates = sorted(prev_rates, reverse=True)
                        m = len(cur_rates)
                        n = len(prev_rates)
                        diff = 0
                        i = 0
                        j = 0
                        while j < n:
                            if i < m and cur_rates[i] > prev_rates[j]:
                                diff += (cur_rates[i]-prev_rates[j])
                                i += 1
                            else:
                                cur_rates.append(prev_rates[j])
                            j += 1
                        rate_dict[prev_ts] = diff
                    else:
                        rate_dict[prev_ts] = sum(cur_rates) - sum(prev_rates)
                    prev_rates = cur_rates
                    cur_rates = [rate]
                prev_ts = timestamp

    fig, ax1 = plt.subplots(1, 1)

    timestamp_axis = [1468501200 + i for i in xrange(3 * 60 * 60)]
    tweet_axis = [response_dict[ts] if ts in response_dict else 0 for ts in timestamp_axis]
    rate_axis = [rate_dict[ts] if ts in rate_dict else 0 for ts in timestamp_axis]
    datetime_axis = [datetime.utcfromtimestamp(ts) for ts in timestamp_axis]

    ax1.plot_date(datetime_axis, tweet_axis, 'r-', ms=2, marker='x')
    ax1.plot_date(datetime_axis, rate_axis, 'g-', ms=2, marker='x')

    xfmt = mdate.DateFormatter('%H:%M:%S')
    ax1.xaxis.set_major_formatter(xfmt)

    ax1.set_xlabel('Jul\' 14 2016 UTC')
    ax1.set_ylabel('Number of receive tweets/out-of-sample tweets')
    ax1.set_title('Figure 1: Secondly receive tweets vs out-of-sample tweets.')

    plt.setp(ax1.get_xticklabels(), rotation=45)

    plt.show()
