#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Script to extract trending videos survival process snapshot.
Time: ~2H30M

Data example:
{country: {vid: [published_at,
                (start_timestamp1, end_timestamp1, last_hours1,
                 viewcount_series1, comment_series1, like_series1, dislike_series1, rank_series1),
                ...],
           ...}}
{'AU': {"kIkoD0pF6_k": ["2017-05-02T22:36:24.000Z",
                        ("2017-05-05-02", "2017-05-06-01", 24,
                         [123, 126, 146, 156, ...], [...], [...], [...], [1, 2, 1, 4, ...]),
                         ...],
        ...}}
"""

from __future__ import division, print_function
import os, json, time
from datetime import datetime, timedelta
import cPickle as pickle


def get_last_hour(time_str1, time_str2):
    # calculate last hours between two timestamp, suppose time_str2 >= time_str1
    FMT = '%Y-%m-%d-%H'
    return (datetime.strptime(time_str2, FMT) - datetime.strptime(time_str1, FMT)).seconds/3600


if __name__ == '__main__':
    # == == == == == == Part 1: Set up experiment parameters == == == == == ==#
    # setting parameters
    prog_start_time = time.time()

    # == == == == == == Part 2: Load data == == == == == == #
    data_loc = '../national-trendings'
    trending_snapshot = {}

    for subdir, _, files in os.walk(data_loc):
        for f in sorted(files):
            country_code = subdir[-2:]
            if country_code not in trending_snapshot:
                trending_snapshot[country_code] = {}
            with open(os.path.join(subdir, f), 'r') as fin:
                for cnt, line in enumerate(fin):
                    try:
                        video = json.loads(line.rstrip())
                    except:
                        continue

                    vid = video['id']
                    timestamp = f[:-4]
                    try:
                        # deal with deleted video
                        published_at = str(video['snippet']['publishedAt'])
                        view_count = int(video['statistics']['viewCount'])
                    except:
                        continue
                    if 'commentCount' in video['statistics']:
                        comment_count = int(video['statistics']['commentCount'])
                    else:
                        comment_count = 'NA'
                    if 'likeCount' in video['statistics']:
                        like_count = int(video['statistics']['likeCount'])
                    else:
                        like_count = 'NA'
                    if 'dislikeCount' in video['statistics']:
                        dislike_count = int(video['statistics']['dislikeCount'])
                    else:
                        dislike_count = 'NA'
                    rank = cnt + 1

                    if vid in trending_snapshot[country_code]:
                        _, end_time, _, _, _, _, _, _ = trending_snapshot[country_code][vid][-1]
                        if get_last_hour(end_time, timestamp) == 1:
                            # modify last elements
                            start_time, end_time, last_hour, views, comments, likes, dislikes, ranks = trending_snapshot[country_code][vid].pop(-1)
                            end_time = timestamp
                            last_hour += 1
                            views.append(view_count)
                            if comments != ['NA']:
                                comments.append(comment_count)
                            if likes != ['NA']:
                                likes.append(like_count)
                            if dislikes != ['NA']:
                                dislikes.append(dislike_count)
                            ranks.append(rank)
                            trending_snapshot[country_code][vid].append((start_time, end_time, last_hour, views, comments, likes, dislikes, ranks))
                        else:
                            # append a new elements
                            trending_snapshot[country_code][vid].append((timestamp, timestamp, 1, [view_count], [comment_count], [like_count], [dislike_count], [rank]))
                    else:
                        trending_snapshot[country_code][vid] = [published_at,
                                                                (timestamp, timestamp, 1, [view_count], [comment_count], [like_count], [dislike_count], [rank])]

    # == == == == == == Part 3: Write to pickle file and print data examples == == == == == == #
    national_trending_pickle = 'national_trendings_snapshot.p'
    if not os.path.exists(national_trending_pickle):
        pickle.dump(trending_snapshot, open(national_trending_pickle, 'wb'))
    # trending_snapshot = pickle.load(open(national_trending_pickle, 'rb'))

    # print example data
    print('Example data in US')
    for vid in trending_snapshot['US'].keys()[:5]:
        print(trending_snapshot['US'][vid])
    print('-'*79)

    print('Example data in AU')
    for vid in trending_snapshot['AU'].keys()[:5]:
        print(trending_snapshot['AU'][vid])
    print('-' * 79)

    print('Example data in JP')
    for vid in trending_snapshot['JP'].keys()[:5]:
        print(trending_snapshot['JP'][vid])
    print('-' * 79)

    # get running time
    print('\n>>> Total running time: {0}'.format(str(timedelta(seconds=time.time() - prog_start_time)))[:-3])
