#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import isodate
from datetime import datetime
import numpy as np
# from nltk.tokenize import RegexpTokenizer
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect


# Extract essential information from each video
# Usage: python extract_video_info.py input_path output_path

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


def extract_info(input_path, truncated=None):
    """
    Extract essential information from each video.
    :param input_path: input file path
    :param truncated: head number of elements extracted
    :return:
    """
    with open(input_path, 'r') as fin:
        for line in fin:
            # skip if data is corrupted
            try:
                video = json.loads(line.rstrip())
            except:
                continue

            # skip if reading duration fails
            try:
                duration = isodate.parse_duration(video['contentDetails']['duration']).seconds
            except Exception as e:
                print(str(e))
                continue

            # skip if not insights data or not watching data
            if 'insights' not in video or video['insights']['avgWatch'] == 'N' or duration == 0:
                continue

            # skip if not topic information
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
                continue

            # skip if not description available or can't determine language
            description = video['snippet']['description']
            try:
                detect_lang = detect(description)
            except:
                continue

            published_at = video['snippet']['publishedAt'][:10]
            start_date = video['insights']['startDate']
            time_diff = (datetime(*map(int, start_date.split('-'))) - datetime(*map(int, published_at.split('-')))).days
            days = read_as_int_array(video['insights']['days'], delimiter=',', truncated=truncated) + time_diff
            days = days[days < truncated]
            daily_view = read_as_int_array(video['insights']['dailyView'], delimiter=',', truncated=len(days))

            # skip if not view in the first num_age day
            if np.sum(daily_view) == 0:
                continue

            id = video['id']
            definition = [0, 1][video['contentDetails']['definition'] == 'hd']
            category_id = video['snippet']['categoryId']
            channel_id = video['snippet']['channelId']

            # title = video['snippet']['title']
            # len_title = len(tokenizer.tokenize(title))
            # polar_title = sid.polarity_scores(title)['compound']
            # len_desc = len(tokenizer.tokenize(description))
            # polar_desc = sid.polarity_scores(description)['compound']

            daily_watch = read_as_float_array(video['insights']['dailyWatch'], delimiter=',', truncated=len(days))

            # fout.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\t{13}\t{14}\n'
            fout.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\n'
                       .format(id, duration, definition,
                               category_id, channel_id, published_at,
                               # len_title, polar_title, len_desc, polar_desc,
                               detect_lang,
                               topics, topics_num,
                               strify(days), strify(daily_view), strify(daily_watch)))


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    age = 120

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    data_loc = sys.argv[1]
    output_loc = sys.argv[2]

    fout = open(output_loc, 'w')
    # fout.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\t{13}\t{14}\n'
    fout.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\n'
               .format('id', 'duration', 'definition',
                       'categoryId', 'channelId', 'publishedAt',
                       # 'titleLength', 'titlePolarity', 'descLength', 'descPolarity',
                       'detectLang',
                       'topics', 'topicsNum',
                       'days', 'dailyView', 'dailyWatch'))

    # tokenizer = RegexpTokenizer(r'\w+')
    # sid = SentimentIntensityAnalyzer()

    if os.path.isdir(data_loc):
        for subdir, _, files in os.walk(data_loc):
            for f in files:
                extract_info(os.path.join(subdir, f), truncated=age)
    else:
        extract_info(data_loc, truncated=age)

    fout.close()