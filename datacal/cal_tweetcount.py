#!/usr/bin/env python

import os
from collections import defaultdict

vid_tc_dict = defaultdict(int)

for subdir, _, files in os.walk('../../data/sampled_may_data/video_stats'):
    for f in sorted(files):
        if f.startswith('2016-05'):
            tweet_date = f[:10]
            print 'tweet date', f[:10]
            filepath = os.path.join(subdir, f)
            with open(filepath, 'r') as datefile:
                for line in datefile:
                    vid, tc = line.rstrip().split(': ')
                    if not tc[0] == '-':
                        vid_tc_dict[vid] += int(tc)

output_file = open('../../data/sampled_may_data/vid_tweetcount.txt', 'w')
for vid, tc in vid_tc_dict.items():
    output_file.write('{0}\t{1}\n'.format(vid, tc))
output_file.close()
