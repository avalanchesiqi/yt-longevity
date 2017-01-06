#!/usr/bin/python

# Usage example:
# python get_rate_limits.py --input='<input_dir>'

import os
import json
import argparse
from datetime import datetime
from collections import defaultdict

BASE_DIR = '../../'


def get_tweet_rate(input_dir, initial_rates, initial_ts):
    tweet_dict = defaultdict(int)
    rate_dict = defaultdict(int)
    prev_ts = initial_ts
    prev_rates = initial_rates
    curr_rates = []

    cnt1 =0
    cnt2 = 0
    ret = []

    for subdir, _, files in os.walk(input_dir):
        for f in sorted(files):
            if f.endswith('txt'):
                with open(os.path.join(subdir, f), 'r') as input_data:
                    for line in input_data:
                        line = line.rstrip()
                        if not line == ',':
                            if line[0] == '[':
                                line = line[1:]
                            elif line[-1] == ']':
                                line = line[:-1]
                            if line:
                                res = json.loads(line)
                                if 'id' in res:
                                    timestamp = int(res['timestamp_ms'][:-3])
                                    tweet_dict[timestamp] += 1
                                else:
                                    if abs(int(res['limit']['timestamp_ms'])%1000 - 500) > 250:
                                        cnt1 += 1
                                    else:
                                        cnt2 += 1
                                        ret.append(int(round(1.0 * int(res['limit']['timestamp_ms']) / 1000)))
                                    timestamp = int(round(1.0 * int(res['limit']['timestamp_ms']) / 1000))
                                    rate = int(res['limit']['track'])
                                    if timestamp == prev_ts:
                                        curr_rates.append(rate)
                                    else:
                                        if len(curr_rates) == 0:
                                            curr_rates.append(rate)
                                        else:
                                            prev_rates = sorted(prev_rates, reverse=True)
                                            curr_rates = sorted(curr_rates, reverse=True)
                                            # each element in curr_rates should be larger than respective prev_rates element
                                            for i in xrange(4):
                                                if i >= len(curr_rates):
                                                    curr_rates.append(prev_rates[i])
                                                elif curr_rates[i] <= prev_rates[i]:
                                                    curr_rates.insert(i, prev_rates[i])
                                            diff = sum(curr_rates) - sum(prev_rates)
                                            if diff < 0:
                                                rate_dict[timestamp - 2] = 0
                                                prev_rates = [0, 0, 0, 0]
                                            else:
                                                print '---------------'
                                                print timestamp
                                                print datetime.utcfromtimestamp(timestamp-2).strftime('%Y-%m-%d %H:%M:%S')
                                                print curr_rates
                                                print prev_rates
                                                print diff
                                                print '***************'
                                                rate_dict[timestamp - 2] = diff
                                                prev_rates = curr_rates
                                            curr_rates = [rate]
                                    prev_ts = timestamp

    print cnt1
    print cnt2
    for i in ret:
        print datetime.utcfromtimestamp(i).strftime('%Y-%m-%d %H:%M:%S')
    return tweet_dict, rate_dict


if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input dir path of video ids, relative to base dir', required=True)
    args = parser.parse_args()

    input_dir = os.path.join(BASE_DIR, args.input)

    initial_rates = [2398706, 2398294, 2397290, 2395588]
    initial_ts = 1483660752
    tweet_dict, rate_dict = get_tweet_rate(input_dir, initial_rates, initial_ts)

    # for t in sorted(rate_dict.keys()):
        # print datetime.utcfromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S'), ":", tweet_dict[t]

    with open(os.path.join(BASE_DIR, 'data/rate_dict.json'), 'w+') as rate_json:
        rate_json.write(json.dumps(rate_dict))
