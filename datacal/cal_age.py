#!/usr/bin/env python

import os
import json
from datetime import datetime
import dateutil.parser
import pytz

BASE_DIR = '../../data/full_jun_data/'


def write_vid_tweetcount_age(dir_path):
    vid_publish = {}
    for subdir, _, files in os.walk(dir_path+'metadata'):
        for f in files:
            if f.endswith('json'):
                filepath = os.path.join(subdir, f)
                with open(filepath, 'r') as filedata:
                    for line in filedata:
                        metadata = json.loads(line.rstrip())
                        try:
                            published_at = metadata['snippet']['publishedAt']
                            # # account for the timezone
                            # utc_published_at = dateutil.parser.parse(published_at)
                            # # convert to AEST
                            # aest_published_at = utc_published_at.replace(tzinfo=pytz.timezone('Australia/ACT'))
                            # vid_publish[metadata['id']] = aest_published_at
                            # not account for th timezone
                            vid_publish[metadata['id']] = published_at[:10]
                        except Exception as exc:
                            print metadata['id'], str(exc)
                            continue
                print f.title(), "is loaded!"

    with open(dir_path+'vid_tweetcount_age.txt', 'w') as output_file:
        for subdir, _, files in os.walk(dir_path+'video_stats'):
            for f in sorted(files):
                tweet_date = f[:10]
                print 'now processing tweet date', tweet_date
                filepath = os.path.join(subdir, f)
                with open(filepath, 'r') as filedata:
                    for line in filedata:
                        vid, tc = line.rstrip().split()
                        if vid in vid_publish:
                            upload_date = vid_publish[vid]
                            # age = (datetime(*map(int, tweet_date.split('-')), tzinfo=pytz.utc)-upload_date).days
                            age = (datetime(*map(int, tweet_date.split('-'))) - datetime(*map(int, upload_date.split('-')))).days
                            output_file.write('{0}\t{1}\t{2}\n'.format(vid, tc, age))


if __name__ == '__main__':
    for i in xrange(2014, 2017):
        write_vid_tweetcount_age('{0}/{1}/'.format(BASE_DIR, i))
