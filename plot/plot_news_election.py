import os
import json
from collections import defaultdict
from datetime import datetime
import dateutil.parser
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates


def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.0fM' % (x * 1e-6)

m_formatter = FuncFormatter(millions)

x_fmt = mdates.DateFormatter('%m-%d %H:%M')

color_dict = {'ABC News': 'r', 'BBC News': 'g', 'The Alex Jones Channel': 'b', 'The Young Turks': 'b', 'RT': 'm'}


if __name__ == '__main__':
    file_loc = '../data/news_elections'
    fig, ax1 = plt.subplots(1, 1)

    for subdir, _, files in os.walk(file_loc):
        for f in files:
            posttime_dict = defaultdict(int)
            filepath = os.path.join(subdir, f)
            with open(filepath, 'r') as filedata:
                for line in filedata:
                    if line.rstrip():
                        video = json.loads(line.rstrip())
                        published_at = video['snippet']['publishedAt']
                        dt = dateutil.parser.parse(published_at)
                        viewcount = int(video['insights']['dailyView'].split(',', 1)[0])
                        channel_title = video['snippet']['channelTitle']
                        if video['liveStreamingDetails']['available']:
                            starttime = dateutil.parser.parse(video['liveStreamingDetails']['actualStartTime'])
                            endtime = dateutil.parser.parse(video['liveStreamingDetails']['actualEndTime'])
                            ax1.plot((starttime, endtime), (viewcount, viewcount), c=color_dict[channel_title], linewidth=1.5)
                        else:
                            posttime_dict[dt] = viewcount

            x_axis = sorted(posttime_dict.keys())
            y_axis = [posttime_dict[t] for t in x_axis]

            ax1.scatter(x_axis, y_axis, s=10, c=color_dict[channel_title], label=channel_title, lw = 0)

    ax1.plot((datetime(2016, 11, 9, 8, 4, 17), datetime(2016, 11, 9, 8, 4)), (0, 8000000), 'y')

    ax1.text(datetime(2016, 11, 9, 8, 0), 6200000, 'President Obama Full Speech on Donald Trump Win')
    ax1.text(datetime(2016, 11, 9, 9, 30), 4000000, 'Hillary Clinton FULL Concession Speech')
    ax1.text(datetime(2016, 11, 9, 0, 0), 3100000, 'Donald Trump Wins US Presidential Election')
    ax1.text(datetime(2016, 11, 9, 4, 0), 2650000, 'Donald Trump VICTORY SPEECH')

    ax1.set_ylim(ymin=0)
    ax1.set_ylim(ymax=8000000)
    ax1.xaxis.set_major_formatter(x_fmt)
    ax1.yaxis.set_major_formatter(m_formatter)
    ax1.set_xlabel('72 hours around US election')
    ax1.set_ylabel('Number of video views in day 1')
    ax1.set_title('Figure 1: News video viewership in 2016 US election')
    ax1.legend(loc='upper left', fontsize=12)

    plt.tight_layout()
    plt.show()
