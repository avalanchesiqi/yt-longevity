#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import sys
import os
import json
import numpy as np
from collections import defaultdict
from scipy import stats
import isodate
from datetime import datetime
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
# warnings.filterwarnings('error')

category_dict = {"42": "Shorts", "29": "Nonprofits & Activism", "24": "Entertainment", "25": "News & Politics", "26": "Howto & Style", "27": "Education", "20": "Gaming", "21": "Videoblogging", "22": "People & Blogs", "23": "Comedy", "44": "Trailers", "28": "Science & Technology", "43": "Shows", "40": "Sci-Fi/Fantasy", "41": "Thriller", "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music", "39": "Horror", "38": "Foreign", "15": "Pets & Animals", "17": "Sports", "19": "Travel & Events", "18": "Short Movies", "31": "Anime/Animation", "30": "Movies", "37": "Family", "36": "Drama", "35": "Documentary", "34": "Comedy", "33": "Classics", "32": "Action/Adventure"}
folder_dict = {"42": "Shorts", "29": "Nonprofits", "24": "Entertainment", "25": "News", "26": "Howto", "27": "Education", "20": "Gaming", "21": "Videoblogging", "22": "People", "23": "Comedy", "44": "Trailers", "28": "Science", "43": "Shows", "40": "SciFi", "41": "Thriller", "1": "Film", "2": "Autos", "10": "Music", "39": "Horror", "38": "Foreign", "15": "Pets", "17": "Sports", "19": "Travel", "18": "ShortMovies", "31": "Anime", "30": "Movies", "37": "Family", "36": "Drama", "35": "Documentary", "34": "Comedy", "33": "Classics", "32": "Action"}


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
    age = 180
    separate = True

    popular_view_percent_matrix = None
    popular_watch_percent_matrix = None
    normal_view_percent_matrix = None
    normal_watch_percent_matrix = None
    unpopular_view_percent_matrix = None
    unpopular_watch_percent_matrix = None

    view_percent_matrix = None
    watch_percent_matrix = None

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    category_id = '25'
    data_loc = '../../data/production_data/random_dataset/{0}.json'.format(category_id)

    # == == == == == == == == Part 3: Read dataset and update matrix == == == == == == == == #
    with open(data_loc, 'r') as filedata:
        for line in filedata:
            video = json.loads(line.rstrip())
            published_at = video['snippet']['publishedAt'][:10]
            start_date = video['insights']['startDate']
            time_diff = (datetime(*map(int, start_date.split('-'))) - datetime(*map(int, published_at.split('-')))).days
            days = read_as_int_array(video['insights']['days'], truncated=age) + time_diff
            days = days[days < age]
            daily_view = read_as_int_array(video['insights']['dailyView'], truncated=len(days))
            total_view = np.sum(daily_view)

            # when view statistic is missing, fill 0s
            filled_view_percent = np.zeros(age)
            filled_view_percent[days] = daily_view/total_view if total_view != 0 else 0

            duration = isodate.parse_duration(video['contentDetails']['duration']).seconds
            daily_watch = read_as_float_array(video['insights']['dailyWatch'], truncated=len(days))
            filled_watch_percent = np.zeros(age)
            filled_watch_percent[days] = np.divide(daily_watch * 60, daily_view * duration, where=(daily_view != 0))
            filled_watch_percent[filled_watch_percent > 1] = 1

            if not separate:
                if view_percent_matrix is None:
                    view_percent_matrix = filled_view_percent
                else:
                    view_percent_matrix = np.vstack((view_percent_matrix, filled_view_percent))
                if watch_percent_matrix is None:
                    watch_percent_matrix = filled_watch_percent
                else:
                    watch_percent_matrix = np.vstack((watch_percent_matrix, filled_watch_percent))
            else:
                if total_view > 1000:
                    if popular_view_percent_matrix is None:
                        popular_view_percent_matrix = filled_view_percent
                    else:
                        popular_view_percent_matrix = np.vstack((popular_view_percent_matrix, filled_view_percent))

                    if popular_watch_percent_matrix is None:
                        popular_watch_percent_matrix = filled_watch_percent
                    else:
                        popular_watch_percent_matrix = np.vstack((popular_watch_percent_matrix, filled_watch_percent))
                elif total_view < 100:
                    if unpopular_view_percent_matrix is None:
                        unpopular_view_percent_matrix = filled_view_percent
                    else:
                        unpopular_view_percent_matrix = np.vstack((unpopular_view_percent_matrix, filled_view_percent))

                    if unpopular_watch_percent_matrix is None:
                        unpopular_watch_percent_matrix = filled_watch_percent
                    else:
                        unpopular_watch_percent_matrix = np.vstack((unpopular_watch_percent_matrix, filled_watch_percent))
                else:
                    if normal_view_percent_matrix is None:
                        normal_view_percent_matrix = filled_view_percent
                    else:
                        normal_view_percent_matrix = np.vstack((normal_view_percent_matrix, filled_view_percent))
                    if normal_watch_percent_matrix is None:
                        normal_watch_percent_matrix = filled_watch_percent
                    else:
                        normal_watch_percent_matrix = np.vstack((normal_watch_percent_matrix, filled_watch_percent))

                # print popular_view_percent_matrix.shape
                # print popular_watch_percent_matrix.shape
                # print normal_view_percent_matrix.shape
                # print normal_watch_percent_matrix.shape
                # print unpopular_view_percent_matrix.shape
                # print unpopular_watch_percent_matrix.shape

    if separate:
        # fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        popular_watch_percent_matrix = np.ma.masked_where(popular_watch_percent_matrix == 0, popular_watch_percent_matrix)
        normal_watch_percent_matrix = np.ma.masked_where(normal_watch_percent_matrix == 0, normal_watch_percent_matrix)
        unpopular_watch_percent_matrix = np.ma.masked_where(unpopular_watch_percent_matrix == 0, unpopular_watch_percent_matrix)

        ax1.plot(np.arange(age), np.median(popular_view_percent_matrix, axis=0), label='popular view percent')
        ax2.plot(np.arange(age), np.ma.median(popular_watch_percent_matrix, axis=0), label='popular watch percent')
        ax1.plot(np.arange(age), np.median(normal_view_percent_matrix, axis=0), label='normal view percent')
        ax2.plot(np.arange(age), np.ma.median(normal_watch_percent_matrix, axis=0), label='normal watch percent')
        ax1.plot(np.arange(age), np.median(unpopular_view_percent_matrix, axis=0), label='unpopular view percent')
        ax2.plot(np.arange(age), np.ma.median(unpopular_watch_percent_matrix, axis=0), label='unpopular watch percent')
        # ax1.boxplot(popular_view_percent_matrix, showmeans=False, showfliers=False, showcaps=False)
        # ax2.boxplot(popular_watch_percent_matrix, showmeans=False, showfliers=False, showcaps=False)
        # ax3.boxplot(normal_view_percent_matrix, showmeans=False, showfliers=False, showcaps=False)
        # ax4.boxplot(normal_watch_percent_matrix, showmeans=False, showfliers=False, showcaps=False)
        # ax5.boxplot(unpopular_view_percent_matrix, showmeans=False, showfliers=False, showcaps=False)
        # ax6.boxplot(unpopular_watch_percent_matrix, showmeans=False, showfliers=False, showcaps=False)

        # for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
        #     ax.set_xlabel('Age')
        # for ax in (ax1, ax3, ax5):
        #     ax.set_ylabel('View Percentage')
        # for ax in (ax2, ax4, ax6):
        #     ax.set_ylabel('Watch Percentage')
        ax1.set_xlabel('Age')
        ax2.set_xlabel('Age')
        ax1.set_ylabel('View Percentage')
        ax2.set_ylabel('Watch Percentage')
        ax1.legend(loc="upper right", fontsize='small')
        ax2.legend(loc="upper right", fontsize='small')
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        watch_percent_matrix = np.ma.masked_where(watch_percent_matrix == 0, watch_percent_matrix)
        ax1.plot(np.arange(age), np.median(view_percent_matrix, axis=0))
        ax2.plot(np.arange(age), np.ma.median(watch_percent_matrix, axis=0))

    # ax1.set_yscale('log')
    # # ax3.set_yscale("log")
    # # ax5.set_yscale("log")
    plt.tight_layout()
    plt.show()
