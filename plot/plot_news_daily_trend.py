import os
import json
from collections import defaultdict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.0fM' % (x * 1e-6)

m_formatter = FuncFormatter(millions)


def thousands(x, pos):
    'The two args are the value and tick position'
    return '%1.0fk' % (x * 1e-3)

k_formatter = FuncFormatter(thousands)


if __name__ == '__main__':
    file_loc = '../data/news'
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    for subdir, _, files in os.walk(file_loc):
        for f in files:
            filepath = os.path.join(subdir, f)
            cnt0 = 0
            dailyview_dict = defaultdict(int)
            cnt1 = 0
            dailyshare_dict = defaultdict(int)
            cnt2 = 0
            dailywatch_dict = defaultdict(float)
            cnt3 = 0
            with open(filepath, 'r') as filedata:
                for line in filedata:
                    if line.rstrip():
                        video = json.loads(line.rstrip())
                        channel_title = video['snippet']['channelTitle']
                        try:
                            start_date = datetime(*map(int, video['insights']['startDate'].split('-')))
                        except Exception, e:
                            cnt0 += 1
                        try:
                            dailyview = map(int, video['insights']['dailyView'].split(','))
                            for i in xrange(len(dailyview)):
                                dailyview_dict[start_date + timedelta(days=i)] += dailyview[i]
                        except Exception, e:
                            cnt1 += 1
                        try:
                            dailyshare = map(int, video['insights']['dailyShare'].split(','))
                            for i in xrange(len(dailyshare)):
                                dailyshare_dict[start_date + timedelta(days=i)] += dailyshare[i]
                        except Exception, e:
                            cnt2 += 1
                        try:
                            dailywatch = map(float, video['insights']['dailyWatch'].split(','))
                            for i in xrange(len(dailywatch)):
                                dailywatch_dict[start_date + timedelta(days=i)] += dailywatch[i]
                        except Exception, e:
                            cnt3 += 1
            x_axis = [datetime(2016, 8, 1, 0, 0) + timedelta(days=i) for i in xrange(120)]
            y1_axis = [dailyview_dict[k] for k in x_axis]
            y2_axis = [dailyshare_dict[k] for k in x_axis]
            y3_axis = [dailywatch_dict[k]/60/24 for k in x_axis]

            y1_label = '{0}: {1:.2f}M'.format(channel_title, np.mean(y1_axis)/1000000)
            y2_label = '{0}: {1:.2f}k'.format(channel_title, np.mean(y2_axis)/1000)
            y3_label = '{0}: {1:.2f}k days'.format(channel_title, np.mean(y3_axis)/1000)

            print 'for {0}, {1} insights, {2} dailyviews, {3} dailyshares, {4} dailywatches not available'.format(channel_title, cnt0, cnt1, cnt2, cnt3)

            ax1.plot_date(x_axis, y1_axis, '-', ms=1, label=y1_label)
            ax2.plot_date(x_axis, y2_axis, '-', ms=1, label=y2_label)
            ax3.plot_date(x_axis, y3_axis, '-', ms=1, label=y3_label)

    ax1.plot((datetime(2016, 11, 9), datetime(2016, 11, 9)), (0, ax1.get_ylim()[1]), 'y--')
    ax2.plot((datetime(2016, 11, 9), datetime(2016, 11, 9)), (0, ax2.get_ylim()[1]), 'y--')
    ax3.plot((datetime(2016, 11, 9), datetime(2016, 11, 9)), (0, ax3.get_ylim()[1]), 'y--')

    ax1.set_ylim(ymin=0)
    ax1.yaxis.set_major_formatter(m_formatter)
    ax1.set_ylabel('Number of views', fontsize=24)
    ax1.set_title('Figure 1: Avg daily view per channel', fontsize=28)
    ax1.legend(loc='upper left', fontsize=24)
    ax1.tick_params(axis='y', labelsize=18)

    ax2.set_ylim(ymin=0)
    ax2.yaxis.set_major_formatter(k_formatter)
    ax2.set_ylabel('Number of shares', fontsize=24)
    ax2.set_title('Figure 2: Avg daily share per channel', fontsize=28)
    ax2.legend(loc='upper left', fontsize=24)
    ax2.tick_params(axis='y', labelsize=18)

    ax3.set_ylim(ymin=0)
    ax3.yaxis.set_major_formatter(k_formatter)
    ax3.set_xlabel('Date', fontsize=24)
    ax3.set_ylabel('Duration of watch time', fontsize=24)
    ax3.set_title('Figure 3: Avg daily watch time per channel', fontsize=28)
    ax3.legend(loc='upper left', fontsize=24)
    ax3.tick_params(axis='x', labelsize=18)
    ax3.tick_params(axis='y', labelsize=18)

    plt.show()
