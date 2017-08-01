#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import isodate
from datetime import datetime
import numpy as np
from langdetect import detect


# Construct tweeted dataset
# Usage: python construct_tweeted_dataset.py input_doc output_doc

def read_as_int_array(content, truncated=None, delimiter=None):
    """
    Read input as an int array.
    :param content: string input
    :param truncated: head number of elements extracted
    :param delimiter: delimiter string
    :return: a numpy int array
    """
    if truncated is None:
        return np.array(map(int, content.split(delimiter)), dtype=np.uint32)
    else:
        return np.array(map(int, content.split(delimiter, truncated)[:-1]), dtype=np.uint32)


def read_as_float_array(content, truncated=None, delimiter=None):
    """
    Read input as a float array.
    :param content: string input
    :param truncated: head number of elements extracted
    :param delimiter: delimiter string
    :return: a numpy float array
    """
    if truncated is None:
        return np.array(map(float, content.split(delimiter)), dtype=np.float64)
    else:
        return np.array(map(float, content.split(delimiter, truncated)[:-1]), dtype=np.float64)


def strify(iterable_struct):
    """
    Convert an iterable structure to comma separated string
    :param iterable_struct: an iterable structure
    :return: a string with comma separated
    """
    return ','.join(map(str, iterable_struct))


def extract_info(input_path, output_path, truncated=None):
    """
    Extract essential information from each video.
    :param input_path: input file path
    :param output_path: output file path
    :param truncated: head number of elements extracted
    :return:
    """
    fout = open(output_path, 'w')
    fout.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\t{13}\t{14}\t{15}\t{16}\t{17}\n'
               .format('id', 'publish', 'duration', 'definition', 'category', 'detect_lang', 'channel',
                       'topics', 'topics_num', 'view@30', 'watch@30', 'wp@30', 'view@120', 'watch@120', 'wp@120',
                       'days', 'daily_view', 'daily_watch'))

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
            # pre-filtering1: upload in jul to aug
            if not (published_at.startswith('2016-07') or published_at.startswith('2016-08')):
                continue

            start_date = video['insights']['startDate']
            time_diff = (datetime(*map(int, start_date.split('-'))) - datetime(*map(int, published_at.split('-')))).days
            days = read_as_int_array(video['insights']['days'], delimiter=',', truncated=truncated) + time_diff
            days = days[days < truncated]
            daily_view = read_as_int_array(video['insights']['dailyView'], delimiter=',', truncated=len(days))
            view_120 = np.sum(daily_view)
            view_30 = np.sum(daily_view[days < 30])

            # pre-filtering2: have at least 100 views in first 30 days
            if view_30 < 100:
                continue

            daily_watch = read_as_float_array(video['insights']['dailyWatch'], delimiter=',', truncated=len(days))
            watch_30 = np.sum(daily_watch[days < 30])
            wp_30 = watch_30*60/view_30/duration
            watch_120 = np.sum(daily_watch)
            wp_120 = watch_120*60/view_120/duration

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
                topics_num = len(topics_set)
            else:
                topics = ''
                topics_num = 0

            # skip if not description available or can't determine language
            description = video['snippet']['description']
            try:
                detect_lang = detect(description)
            except:
                detect_lang = 'NA'

            vid = video['id']
            definition = [0, 1][video['contentDetails']['definition'] == 'hd']
            category = video['snippet']['categoryId']
            channel = video['snippet']['channelId']

            fout.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\t{13}\t{14}\t{15}\t{16}\t'
                       '{17}\n'.format(vid, published_at, duration, definition, category, detect_lang, channel, topics,
                                       topics_num, view_30, watch_30, wp_30, view_120, watch_120, wp_120, strify(days),
                                       strify(daily_view), strify(daily_watch)))
    fout.close()


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    age = 120
    input_loc = sys.argv[1]
    output_loc = sys.argv[2]
    if not os.path.exists(output_loc):
        os.mkdir(output_loc)

    # == == == == == == == == Part 2: Construct dataset == == == == == == == == #
    for subdir, _, files in os.walk(input_loc):
        for f in files:
            extract_info(os.path.join(subdir, f), os.path.join(output_loc, f[:-4]+'txt'), truncated=age)
