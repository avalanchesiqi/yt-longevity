#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import numpy as np
import isodate
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

category_dict = {"42": "Shorts", "29": "Nonprofits & Activism", "24": "Entertainment", "25": "News & Politics", "26": "Howto & Style", "27": "Education", "20": "Gaming", "21": "Videoblogging", "22": "People & Blogs", "23": "Comedy", "44": "Trailers", "28": "Science & Technology", "43": "Shows", "40": "Sci-Fi/Fantasy", "41": "Thriller", "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music", "39": "Horror", "38": "Foreign", "15": "Pets & Animals", "17": "Sports", "19": "Travel & Events", "18": "Short Movies", "31": "Anime/Animation", "30": "Movies", "37": "Family", "36": "Drama", "35": "Documentary", "34": "Comedy", "33": "Classics", "32": "Action/Adventure"}
folder_dict = {"42": "Shorts", "29": "Nonprofits", "24": "Entertainment", "25": "News", "26": "Howto", "27": "Education", "20": "Gaming", "21": "Videoblogging", "22": "People", "23": "Comedy", "44": "Trailers", "28": "Science", "43": "Shows", "40": "SciFi", "41": "Thriller", "1": "Film", "2": "Autos", "10": "Music", "39": "Horror", "38": "Foreign", "15": "Pets", "17": "Sports", "19": "Travel", "18": "ShortMovies", "31": "Anime", "30": "Movies", "37": "Family", "36": "Drama", "35": "Documentary", "34": "Comedy", "33": "Classics", "32": "Action"}


def read_as_int_array(content):
    return np.array(map(int, content.split(',')), dtype=np.uint32)


def read_as_float_array(content):
    return np.array(map(float, content.split(',')), dtype=np.float64)


def safe_div(a, b, duration):
    c = []
    for aa, bb in zip(a, b):
        if bb == 0:
            c.append(0)
        else:
            if aa * 60 / bb / duration > 1:
                c.append(1)
            else:
                c.append(aa * 60 / bb / duration)
    return np.array(c)


def is_valid_video(arr):
    m = len(arr)
    n = sum([1 for x in arr if x < 100])
    if n > m/2 or len(arr) < 8:
        return False
    return True


def get_lin_coef(arr):
    lr = LinearRegression()
    x = np.arange(1, len(arr) + 1).reshape(-1, 1)
    y = np.array(arr).reshape(-1, 1)
    lr.fit(x, y)
    y_pred = lr.predict(x)
    return lr.coef_[0][0], lr.intercept_[0], np.sqrt(metrics.mean_squared_error(y, y_pred)), metrics.r2_score(y, y_pred), metrics.mean_absolute_error(y, y_pred)


def plot_data(watch_percent, weekly_view, weekly_watch, video, reg_text):
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    x_axis = np.arange(1, len(watch_percent) + 1)
    lns1 = ax1.plot(x_axis, weekly_view, 'D-', c='r', ms=3, mfc='None', mec='r', mew=1, label='weekly viewership')
    # lns2 = ax2.plot(x_axis, watch_percent, 'o-', c='b', ms=3, mfc='None', mec='b', mew=1, label='weekly watch percentage')

    coef, intercept, rmse, r2, std_err = get_lin_coef(watch_percent)
    # lns3 = ax2.plot(x_axis, [coef*x+intercept for x in x_axis], 's--', c='g', ms=3, mfc='None', mec='g', mew=1, label='fitted watch percentage')
    sns.regplot(x=x_axis, y=watch_percent, ci=95)
    # ax2.plot(x_axis, [coef * x + intercept - 2*rmse for x in x_axis], '--', c='g')
    # ax2.plot(x_axis, [coef * x + intercept + 2*rmse for x in x_axis], '--', c='g')

    # # precision of prediction
    # within = 0
    # for x in x_axis:
    #     if coef * x + intercept - 2*rmse <= watch_percent[x-1] <= coef * x + intercept + 2*rmse:
    #         within += 1
    # prec = 100.0*within/len(watch_percent)

    ax1.set_xlim(xmin=0)
    ax1.set_xlim(xmax=(week_num+1)*7)
    ax1.set_ylim(ymin=0)
    ax2.set_ylim(ymin=0)
    ax2.set_ylim(ymax=1)
    ax1.set_xlabel('video age (day)')
    ax1.set_ylabel('view', color='r')
    ax1.tick_params('y', colors='r')
    ax2.set_ylabel('watch percentage', color='b')
    ax2.tick_params('y', colors='b')

    title = video['snippet']['title'].encode('ascii', 'ignore').decode('ascii')
    vid = video['id']
    duration = video['contentDetails']['duration']
    category = category_dict[video['snippet']['categoryId']]
    ax2.text(3, 0.83, '{0}\n{1}, {2}\n{3}, {4:.2f}d\n{5}'
             .format(vid, category, duration, sum(weekly_view[:8]), sum(weekly_watch[:8])/60/24, reg_text),
             bbox={'facecolor': 'green', 'alpha': 0.5})

    # lns = lns1 + lns2 + lns3
    # labs = [l.get_label() for l in lns]
    # plt.legend(lns, labs, loc='upper right', fontsize='small')
    fig.savefig(os.path.join(output_loc, video['id']))
    # plt.show()
    plt.clf()


def update_matrix(filepath):
    with open(filepath, 'r') as filedata:
        for line in filedata:
            video = json.loads(line.rstrip())
            if video['insights']['dailyWatch'] == 'N':
                continue
            duration = isodate.parse_duration(video['contentDetails']['duration']).seconds
            if duration > 0:
                published_at = video['snippet']['publishedAt'][:10]
                start_date = video['insights']['startDate']
                time_diff = (datetime(*map(int, start_date.split('-'))) - datetime(*map(int, published_at.split('-')))).days
                ages = read_as_int_array(video['insights']['days']) + time_diff
                daily_view = read_as_int_array(video['insights']['dailyView'])
                daily_watch = read_as_float_array(video['insights']['dailyWatch'])

                get_weekly = False
                if get_weekly:
                    weekly_view = []
                    weekly_watch = []
                    # get seven days data
                    for week_idx in np.arange(week_num):
                        valid_idx = (ages >= 7 * week_idx) * (ages < 7 * (week_idx + 1))
                        if np.sum(daily_view[valid_idx]) > 0:
                            weekly_view.append(np.sum(daily_view[valid_idx]))
                            weekly_watch.append(np.sum(daily_watch[valid_idx]))

                    if is_valid_video(weekly_view):
                        watch_percent = safe_div(weekly_watch, weekly_view, duration)

                        coef, intercept, mse, r2, std_err = get_lin_coef(watch_percent)
                        output_stats_file.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(video['id'], coef, intercept, mse, r2, std_err))

                        reg_text = '{0:.5f}, {1:.5f}, {2:.5f}, {3:.5f}, {4:.5f}'.format(coef, intercept, mse, r2, std_err)
                        plot_data(watch_percent, weekly_view, weekly_watch, video, reg_text)
                else:
                    if is_valid_video(daily_view):
                        watch_percent = safe_div(daily_watch, daily_view, duration)

                        coef, intercept, mse, r2, std_err = get_lin_coef(watch_percent)
                        output_stats_file.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(video['id'], coef, intercept, mse, r2, std_err))

                        reg_text = '{0:.5f}, {1:.5f}, {2:.5f}, {3:.5f}, {4:.5f}'.format(coef, intercept, mse, r2, std_err)
                        plot_data(watch_percent, daily_view, daily_watch, video, reg_text)


if __name__ == '__main__':
    category_id = '43'
    data_loc = '../../data/byCategory/{0}.json'.format(category_id)
    output_loc = '../linux_figs/detailed_videos/{0}'.format(folder_dict[category_id])

    # make output dir if not exists, otherwise skip the program
    # if os.path.exists(output_loc):
    #     print 'output directory already exists! change output dir...'
    #     sys.exit(1)
    # else:
    #     os.makedirs(output_loc)

    output_stats_file = open('../linux_figs/detailed_videos/{0}_stats.txt'.format(folder_dict[category_id]), 'w')
    output_stats_file.write('video_id\tcoef\tintercept\tmse\tr2\tstd_err\n')
    week_num = 52

    fig = plt.figure()

    if os.path.isdir(data_loc):
        for subdir, _, files in os.walk(data_loc):
            for f in files:
                filepath = os.path.join(subdir, f)
                update_matrix(filepath)
    else:
        update_matrix(data_loc)

    output_stats_file.close()
