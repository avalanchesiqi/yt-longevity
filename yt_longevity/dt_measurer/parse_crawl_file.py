#!/usr/bin/python

# Usage example:
# python parse_crawl_file.py --input='<input_file>' --output='<output_file>' --start='<start_timestamp>' --end='<end_timestamp>'

import os
import argparse
import re
import json

BASE_DIR = '../../'


def _extract_vid_from_expanded_url(expanded_url):
        if 'watch?' in expanded_url and 'v=' in expanded_url:
            vid = expanded_url.split('v=')[1][:11]
        elif 'youtu.be' in expanded_url:
            vid = expanded_url.rsplit('/', 1)[-1][:11]
        else:
            return None
        # valid condition: contains only alphanumeric, dash or underline
        valid = re.match('^[\w-]+$', vid) is not None
        if valid and len(vid) == 11:
            return vid
        return None


def _extract_vids(tweet):
    urls = []
    if 'entities' in tweet.keys() and 'urls' in tweet['entities']:
        urls.extend(tweet['entities']['urls'])
    if 'retweeted_status' in tweet.keys():
        if 'entities' in tweet['retweeted_status'] and 'urls' in tweet['retweeted_status']['entities']:
            urls.extend(tweet['retweeted_status']['entities']['urls'])
        if 'extended_tweet' in tweet['retweeted_status']:
            if 'entities' in tweet['retweeted_status']['extended_tweet'] and 'urls' in tweet['retweeted_status']['extended_tweet']['entities']:
                urls.extend(tweet['retweeted_status']['extended_tweet']['entities']['urls'])
    if 'quoted_status' in tweet.keys():
        if 'entities' in tweet['quoted_status'] and 'urls' in tweet['quoted_status']['entities']:
            urls.extend(tweet['quoted_status']['entities']['urls'])
    vids = set()
    for url in urls:
        if url['expanded_url'] is not None:
            vid = _extract_vid_from_expanded_url(url['expanded_url'])
            if vid is not None:
                vids.add(vid)
    return vids


if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input dir path of video ids, relative to base dir', required=True)
    parser.add_argument('-o', '--output', help='output file path of video data, relative to base dir', required=True)
    parser.add_argument('-s', '--start', type=int, help='start timestamp of target interval', required=True)
    parser.add_argument('-e', '--end', type=int, help='end timestamp of target interval', required=True)
    args = parser.parse_args()

    input_dir = os.path.join(BASE_DIR, args.input)
    output_path = os.path.join(BASE_DIR, args.output)
    start = args.start
    end = args.end

    output_data = open(output_path, 'a+')
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
                                tweet = json.loads(line)

                                if 'id' in tweet:
                                    timestamp_ms = int(tweet['timestamp_ms'])
                                    created_at = tweet['created_at']
                                    id = tweet['id']
                                    vids = _extract_vids(tweet)

                                    if start < timestamp_ms < end:
                                        output_data.write('{0}: {1}\n'.format('id', id))
                                        if vids is not None and not len(vids) == 0:
                                            for vid in vids:
                                                output_data.write('{0}: {1}\n'.format('video_id', vid))
                                        output_data.write('{0}: {1}\n'.format('posted_time', created_at))
                                        output_data.write('----------\n')
    output_data.close()
