#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Scripts to construct quality dataset from raw json collection.

Usage: python construct_quality_dataset.py input_doc output_filename
Example: python construct_quality_dataset.py ../../production_data/quality_dataset/vevo ../../production_data/quality_dataset/vevo.txt
Time: ~15M
"""

from __future__ import division, print_function
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import json, isodate, time
from datetime import datetime, timedelta
import cPickle as pickle
import numpy as np
from langdetect import detect

from utils.helper import read_as_float_array, read_as_int_array, strify
from utils.converter import to_relative_engagement


def extract_info(input_path, output_file, truncated=None):
    """
    Extract essential information from each video.
    :param input_path: input file path
    :param output_file: output file handler
    :param truncated: head number of elements extracted
    :return:
    """
    with open(input_path, 'r') as fin:
        for line in fin:
            # skip if data is corrupted or reading duration fails
            try:
                video = json.loads(line.rstrip())
                duration = isodate.parse_duration(video['contentDetails']['duration']).seconds
            except:
                continue

            # skip if not insights data or not watching data
            if 'insights' not in video or video['insights']['avgWatch'] == 'N' or duration == 0:
                continue

            published_at = video['snippet']['publishedAt'][:10]
            start_date = video['insights']['startDate']
            time_diff = (datetime(*map(int, start_date.split('-'))) - datetime(*map(int, published_at.split('-')))).days
            days = read_as_int_array(video['insights']['days'], delimiter=',', truncated=truncated) + time_diff
            days = days[days < truncated]
            daily_view = read_as_int_array(video['insights']['dailyView'], delimiter=',', truncated=len(days))
            view30 = np.sum(daily_view[days < 30])

            # pre-filtering: have at least 100 views in first 30 days
            if view30 < 100:
                continue

            daily_watch = read_as_float_array(video['insights']['dailyWatch'], delimiter=',', truncated=len(days))
            watch30 = np.sum(daily_watch[days < 30])
            wp30 = watch30*60/view30/duration
            # upper bound watch percentage to 1
            if wp30 > 1:
                wp30 = 1
            re30 = to_relative_engagement(engagement_map, duration, wp30, lookup_keys=lookup_durations)

            # topic information
            if 'topicDetails' in video:
                if 'topicIds' in video['topicDetails']:
                    topic_ids = set(video['topicDetails']['topicIds'])
                else:
                    topic_ids = set()
                if 'relevantTopicIds' in video['topicDetails']:
                    relevant_topic_ids = set(video['topicDetails']['relevantTopicIds'])
                else:
                    relevant_topic_ids = set()
                topics_set = topic_ids.union(relevant_topic_ids)
                topics = strify(topics_set)
            else:
                topics = 'NA'

            # detect description language
            description = video['snippet']['description']
            try:
                detect_lang = detect(description)
            except:
                detect_lang = 'NA'

            vid = video['id']
            definition = [0, 1][video['contentDetails']['definition'] == 'hd']
            category = video['snippet']['categoryId']
            channel = video['snippet']['channelId']

            output_file.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\t{13}\t{14}\n'
                              .format(vid, published_at, duration, definition, category, detect_lang, channel, topics,
                                      view30, watch30, wp30, re30, strify(days), strify(daily_view), strify(daily_watch)))


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    print('>>> Start to extract video to target file...')
    start_time = time.time()

    engagement_map_loc = '../data_preprocess/engagement_map.p'
    if not os.path.exists(engagement_map_loc):
        print('Engagement map not generated, start with generating engagement map first in ../data_preprocess dir!.')
        print('Exit program...')
        sys.exit(1)

    engagement_map = pickle.load(open(engagement_map_loc, 'rb'))
    lookup_durations = np.array(engagement_map['duration'])

    age = 120
    input_loc = sys.argv[1]
    output_path = sys.argv[2]

    fout = open(output_path, 'w')
    fout.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\t{13}\t{14}\n'
               .format('id', 'publish', 'duration', 'definition', 'category', 'detect_lang', 'channel', 'topics',
                       'view@30', 'watch@30', 'wp@30', 're@30', 'days', 'daily_view', 'daily_watch'))

    # == == == == == == == == Part 2: Construct dataset == == == == == == == == #
    for subdir, _, files in os.walk(input_loc):
        for f in files:
            extract_info(os.path.join(subdir, f), fout, truncated=age)
            print('>>> Finish converting file {0}!'.format(os.path.join(subdir, f)))

    fout.close()

    # get running time
    print('\n>>> Total running time: {0}'.format(str(timedelta(seconds=time.time() - start_time)))[:-3])
