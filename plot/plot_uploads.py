#!/usr/bin/env python
"""
Show how to make date plots in matplotlib using date tick locators and
formatters.  See major_minor_demo1.py for more information on
controlling major and minor ticks

All matplotlib date plotting is done by converting date instances into
days since the 0001-01-01 UTC.  The conversion, tick locating and
formatting is done behind the scenes so this is most transparent to
you.  The dates module provides several converter functions date2num
and num2date

This example requires an active internet connection since it uses
yahoo finance to get the data for plotting
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from datetime import date, timedelta
import json

CATEGORIES = ["Travel & Events", "Trailers", "Sports", "Shows", "Science & Technology", "Pets & Animals",
              "People & Blogs", "Nonprofits & Activism", "News & Politics", "Music", "Movies", "Howto & Style",
              "Gaming", "Film & Animation", "Entertainment", "Education", "Comedy", "Autos & Vehicles"]
STARTDATE = date(2014, 6, 1)
ENDDATE = date(2016, 6, 30)
BREAKDATE = date(2015, 2, 6)


def aggregate_data(filepath, video_series, category_dict):
    with open(filepath, 'r') as datafile:
        n = len(video_series)
        for line in datafile:
            doc = json.loads(line.strip())
            if 'snippet' in doc['value']['metadata']:
                if ('categoryTitle' in doc['value']['metadata']['snippet']
                    and 'publishedAt' in doc['value']['metadata']['snippet']
                    and '$date' in doc['value']['metadata']['snippet']['publishedAt']):
                    upload_date = doc['value']['metadata']['snippet']['publishedAt']['$date'][:10]
                    y, m, d = map(int, upload_date.split('-'))
                    delta = (date(y, m, d)-STARTDATE).days
                    if 0 <= delta < n:
                        if np.isnan(video_series[delta]):
                            continue
                        else:
                            category = doc['value']['metadata']['snippet']['categoryTitle']
                            video_series[delta] += 1
                            category_dict[category][delta] += 1


def load_daily_views(input_dir):
    print "\nStart aggregating daily viewcounts..."

    def get_missing_days():
        missing_days = []

        def add_missing_day(start, end):
            missing_days.extend([i for i in xrange((start - STARTDATE).days, (end - STARTDATE).days + 1, 1)])

        add_missing_day(date(2014, 12, 24), date(2015, 1, 5))
        add_missing_day(date(2015, 7, 29), date(2015, 9, 12))
        add_missing_day(date(2015, 10, 1), date(2015, 10, 5))
        add_missing_day(date(2015, 10, 19), date(2015, 11, 5))
        add_missing_day(date(2016, 1, 20), date(2016, 3, 15))
        return missing_days

    missing_days = get_missing_days()

    n = (ENDDATE-STARTDATE).days
    video_series = [np.nan if i in missing_days else 0 for i in xrange(n + 1)]
    category_dict = {}
    for category in CATEGORIES:
        category_dict[category] = [np.nan if i in missing_days else 0 for i in xrange(n + 1)]

    for i in xrange(10):
        filepath = "{0}/videoMetadata_{1}.json".format(input_dir, i)
        aggregate_data(filepath, video_series, category_dict)

    print "\nFinish aggregating daily viewcounts.\n"
    return video_series, category_dict


def nice_format(ax):
    # every 3rd month
    months = MonthLocator(range(1, 13), bymonthday=1, interval=3)
    months_fmt = DateFormatter("%b '%y")

    # format the ticks
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(months_fmt)
    ax.autoscale_view()

    # format the coords message box
    def view(x):
        return '{0}'.format(x)

    ax.fmt_xdata = DateFormatter('%Y-%m-%d')
    ax.fmt_ydata = view
    ax.grid(True)

    ax.xaxis_date()
    ax.legend(loc='best')


def curve_fit(arr, ax, c='b'):
    n = len(arr)
    train_x = np.arange(1, n+1, 1)
    train_y = np.array(arr)

    idx = np.isfinite(train_y)
    z = np.polyfit(train_x[idx], train_y[idx], 6)
    f = np.poly1d(z)

    x_new = np.linspace(train_x[0], train_x[-1], 1000)
    y_new = f(x_new)
    dates_new = [STARTDATE + timedelta(days=i) for i in x_new]

    ax.plot_date(dates_new, y_new, linestyle='-', marker='', color=c)


def draw_uploads(video_series, category_dict, *args):

    n = (ENDDATE - STARTDATE).days
    dates = [STARTDATE + timedelta(days=i) for i in xrange(n + 1)]

    fig, (ax1, ax2) = plt.subplots(2, 1)

    # CATEGORIES = ["Travel & Events", "Trailers", "Sports", "Shows", "Science & Technology", "Pets & Animals",
    #               "People & Blogs", "Nonprofits & Activism", "News & Politics", "Music", "Movies", "Howto & Style",
    #               "Gaming", "Film & Animation", "Entertainment", "Education", "Comedy", "Autos & Vehicles"]

    nb = (BREAKDATE - STARTDATE).days
    lowest = np.nanargmin(video_series[nb:])

    feb_onwards = dates[nb:]
    feb_onwards_vs_increment = np.array([v - video_series[nb+lowest] for v in video_series[nb:]])
    feb_onwards_pb_increment = np.array([v - category_dict['People & Blogs'][nb+lowest] for v in category_dict['People & Blogs'][nb:]])
    feb_onwards_ga_increment = np.array([v - category_dict['Gaming'][nb+lowest] for v in category_dict['Gaming'][nb:]])
    feb_onwards_et_increment = np.array([v - category_dict['Entertainment'][nb+lowest] for v in category_dict['Entertainment'][nb:]])
    feb_onwards_mu_increment = np.array([v - category_dict['News & Politics'][nb+lowest] for v in category_dict['News & Politics'][nb:]])

    # video generation trend
    ax1.plot_date(feb_onwards, feb_onwards_vs_increment, 'k-', label="video generation trend")
    # ax1.plot_date(dates[nb:], [video_series[i]-category_dict['Gaming'][i]-category_dict['People & Blogs'][i] for i in xrange(nb, n+1)], 'r-', label="video generation trend without Gaming")
    ax1.plot_date(feb_onwards, feb_onwards_mu_increment+feb_onwards_et_increment + feb_onwards_pb_increment + feb_onwards_ga_increment, 'r-', label="PB&GA&ET&NP video generation trend")
    # ax1.plot_date(feb_onwards, feb_onwards_et_increment+feb_onwards_pb_increment+feb_onwards_ga_increment, 'r-', label="PB&GA&ET video generation trend")
    ax1.plot_date(feb_onwards, feb_onwards_pb_increment+feb_onwards_ga_increment, 'g-', label="PB&GA video generation trend")
    ax1.plot_date(feb_onwards, feb_onwards_ga_increment, 'b-', label="Gaming video generation trend")
    ax1.bar(feb_onwards, np.isnan(feb_onwards_vs_increment)*600, color=(.9, .9, .9))
    # ax1.axvline(date(2015, 2, 6), color='k', zorder=0)
    # ax1.annotate('6 Feb 2015\nsuspicious drop date', xy=(date(2015, 2, 6), 400), xytext=(date(2015, 3, 12), 400), arrowprops=dict(facecolor='black', shrink=0.05))
    # curve_fit(video_series[nb:], ax1)
    nice_format(ax1)

    # ax2.plot_date(feb_onwards, 100.0*(feb_onwards_mu_increment+feb_onwards_et_increment+feb_onwards_pb_increment+feb_onwards_ga_increment)/feb_onwards_vs_increment, 'b-')
    ax2.plot_date(feb_onwards, 100.0*(feb_onwards_mu_increment+feb_onwards_et_increment + feb_onwards_pb_increment + feb_onwards_ga_increment)/feb_onwards_vs_increment, 'r-')
    ax2.plot_date(feb_onwards, 100.0*(feb_onwards_pb_increment+feb_onwards_ga_increment)/feb_onwards_vs_increment, 'g-')
    ax2.plot_date(feb_onwards, 100.0*feb_onwards_ga_increment/feb_onwards_vs_increment, 'b-')
    ax2.bar(feb_onwards, np.isnan(feb_onwards_vs_increment) * 100, color=(.9, .9, .9))
    ax2.set_ylim(0, 100)

    # # category video percentage
    # colorcycle = 'rbgmyck'
    # for i in xrange(len(args)):
    #     category_ratio = [100.0 * category_dict[args[i]][j] / video_series[j] for j in xrange(n+1)]
    #     ax2.plot_date(dates, category_ratio, linestyle='-', marker='', color=colorcycle[i], label=args[i])
    #     curve_fit(category_ratio, ax2, c=colorcycle[i])
    # ymin, ymax = ax2.get_ylim()
    # ax2.bar(dates, np.isnan(video_series)*ymax, color=(.9, .9, .9))
    # ylabels = ["{0}%".format(i) for i in xrange(int(ymin), int(ymax+1), 5)]
    # ax2.set_yticklabels(ylabels, fontsize='medium')
    # nice_format(ax2)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    video_series, category_dict = load_daily_views("../datasets/metadata/")
    draw_uploads(video_series, category_dict, "People & Blogs", "Gaming", "Music", "News & Politics")
