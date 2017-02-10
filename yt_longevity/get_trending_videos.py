#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Usage example:
# python get_trending_videos.py -o output

import sys
import os
import argparse
import json
from datetime import datetime
from apiclient import discovery


API_SERVICE = 'youtube'
API_VERSION = 'v3'
DEVELOPER_KEY = 'AIzaSyBxNpscfnZ5-we_4-PfGEB4LIadRYOjs-M'
BASE_DIR = '../'


def get_region_trending_videos(output_filename, region_code):
    trending_videos = None
    # Call the videos().list method to retrieve trending videos in designated region.
    try:
        responses = youtube.videos().list(part='snippet,statistics,topicDetails', chart='mostPopular', maxResults=50,
                                          regionCode=region_code).execute()
        trending_videos = responses["items"]
    except Exception as e:
        print >> sys.stderr, 'fail to crawl trending videos with error {0}'.format(str(e))

    if trending_videos is not None:
        with open(output_filename, 'w+') as output_file:
            for video in trending_videos:
                output_file.write('{}\n'.format(json.dumps(video)))


def get_trending_videos_api(output_dir):
    """Finds trending videos in all regions."""

    region_list = []
    with open(BASE_DIR+'conf/region_list.txt', 'r') as regions:
        for line in regions:
            rcode = line.rstrip().split()[0]
            region_list.append(rcode)

    for region_code in region_list:
        region_dir = os.path.join(output_dir, region_code)
        if not os.path.exists(region_dir):
            os.makedirs(region_dir)
        output_filename = os.path.join(region_dir, output_name)
        get_region_trending_videos(output_filename, region_code)


if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', help='output dir of results', required=True)
    args = parser.parse_args()

    output_name = datetime.strftime(datetime.utcnow(), "%Y-%m-%d-%H")+'.txt'
    youtube = discovery.build(API_SERVICE, API_VERSION, developerKey=DEVELOPER_KEY)
    get_trending_videos_api(args.o)
