import os
import numpy as np
import json
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdate


def extract_timestamp(obj):
    return int(json.loads(obj)['limit']['timestamp_ms'])

if __name__ == '__main__':
    file_loc = '../../data/twitter-sampling'
    filename = '2016_06_19_17.txt'
    filepath = os.path.join(file_loc, filename)

    ts_arr = []
    with open(filepath, 'r') as filedata:
        for line in filedata:
            msg = json.loads(line.rstrip())
            ts = int(round(1.0 * int(msg['limit']['timestamp_ms']) / 1000))
            ts = datetime.utcfromtimestamp(ts)
            ts_arr.append(ts)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(np.arange(1, len(ts_arr)+1), ts_arr, marker='o', s=1, c='b')
    yfmt = mdate.DateFormatter('%H:%M:%S')
    ax1.yaxis.set_major_formatter(yfmt)

    ax1.set_xlim(xmin=0)
    ax1.set_ylim(ymin=min(ts_arr))
    ax1.set_ylim(ymax=max(ts_arr))
    ax1.set_xlabel('Streaming order')
    ax1.set_ylabel('Timestamp')
    ax1.set_title('Figure 1: Rate limit messages timestamp wrt streaming order.', y=-0.12)

    ts_dict = defaultdict(int)
    for ts in ts_arr:
        ts_dict[ts] += 1

    freq_dict = defaultdict(int)
    for freq in ts_dict.values():
        freq_dict[freq] += 1

    ax2.bar(freq_dict.keys(), freq_dict.values(), width=0.5, align='center')
    ax2.set_xlabel('Rate limit message number')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Figure 2: Distribution of rate limit messages per nearby second', y=-0.12)

    plt.show()
