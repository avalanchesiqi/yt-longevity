#!/usr/bin/env python

import os

vid_set = set()

for subdir, _, files in os.walk('../../data/sampled_may_data/video_ids'):
    for f in sorted(files):
        if f.startswith('2016-05'):
            tweet_date = f[:10]
            print 'tweet date', f[:10]
            filepath = os.path.join(subdir, f)
            with open(filepath, 'r') as datefile:
                for line in datefile:
                    vid = line.rstrip()
                    vid_set.add(vid)

output_file = open('../../data/sampled_may_data/may_2016_uniq_vids.txt', 'w')
for vid in vid_set:
    output_file.write('{0}\n'.format(vid))
output_file.close()
