#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import isodate
from datetime import datetime
from scipy import stats
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def read_as_int_array(content, truncated=None):
    if truncated is None:
        return np.array(map(int, content.split(',')), dtype=np.uint32)
    else:
        return np.array(map(int, content.split(',')), dtype=np.uint32)[:truncated]


def read_as_float_array(content, truncated=None):
    if truncated is None:
        return np.array(map(float, content.split(',')), dtype=np.float64)
    else:
        return np.array(map(float, content.split(',')), dtype=np.float64)[:truncated]


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    age = 30

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    category_id = '43'
    data_loc = '../../data/production_data/random_dataset/{0}.json'.format(category_id)

    sid = SentimentIntensityAnalyzer()

    i = 0

    with open(data_loc, 'r') as fin:
        for line in fin:
            video = json.loads(line.rstrip())
            id = video['id']
            title = video['snippet']['title']
            description = video['snippet']['description']
            print id
            title_tokens = nltk.word_tokenize(title)
            # print title
            # print title_tokens
            print title
            # print title_tokens
            print len(title_tokens), sid.polarity_scores(title)['compound']

            description_tokens = nltk.word_tokenize(description)
            # print description
            # print description_tokens
            print len(description_tokens), sid.polarity_scores(description)['compound']

            published_at = video['snippet']['publishedAt'][:10]
            start_date = video['insights']['startDate']
            time_diff = (datetime(*map(int, start_date.split('-'))) - datetime(*map(int, published_at.split('-')))).days
            days = read_as_int_array(video['insights']['days'], truncated=age) + time_diff
            days = days[days < age]
            daily_view = read_as_int_array(video['insights']['dailyView'], truncated=len(days))
            total_view = np.sum(daily_view)

            # when view statistic is missing, fill 0s
            filled_view_percent = np.zeros(age)
            filled_view_percent[days] = daily_view / total_view if total_view != 0 else 0

            duration = isodate.parse_duration(video['contentDetails']['duration']).seconds
            daily_watch = read_as_float_array(video['insights']['dailyWatch'], truncated=len(days))
            filled_watch_percent = np.zeros(age)
            filled_watch_percent[days] = np.divide(daily_watch * 60, daily_view * duration, where=(daily_view != 0))
            filled_watch_percent[filled_watch_percent > 1] = 1

            print float(video['insights']['avgWatch']) * 60 / duration

            r, p = stats.pearsonr(daily_watch*60/daily_view/duration, daily_view)
            print r, p

            if p < 0.05:
                print daily_watch*60/daily_view/duration
                print daily_view

            i += 1
            if i > 10:
                break
