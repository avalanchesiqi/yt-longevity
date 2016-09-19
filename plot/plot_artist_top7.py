#!/usr/bin/env python

import sys
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

COLOR_LIST = ['r', 'c', 'k', 'g', 'y', 'm', 'b']
NAME_MAP = {'bruno_mars': 'Bruno Mars', 'taylor_swift': 'Taylor Swift', 'justin_bieber': 'Justin Bieber',
            'katy_perry': 'Katy Perry', 'rihanna': 'Rihanna', 'selena_gomez': 'Selena Gomez', 'adele': 'Adele',
            'one_direction': 'One Direction', 'one_republic': 'One Republic', 'imagine_dragons': 'Imagine Dragons'}

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)


dir_path = sys.argv[1]
artist_name = NAME_MAP[dir_path.rsplit('/', 1)[1]]
cnt = 0


def thousands(x, pos):
    return '{0:.1f}K'.format(x/1000)

k_formatter = FuncFormatter(thousands)

for subdir, _, files in os.walk(dir_path):
    for f in files:
        c = COLOR_LIST[cnt]
        file_path = os.path.join(subdir, f)
        with open(file_path, 'r') as data:
            start_date, dailyview, totalview, dailyshare, totalshare = data.readline().rstrip().split()[:5]
            y, m, d = map(int, start_date.split('-'))
            start_date = datetime(y, m, d)
            ax1_y = map(int, dailyview.split(','))
            ax1_x = [start_date + timedelta(days=i) for i in xrange(len(ax1_y))]
            ax1.plot_date(ax1_x, ax1_y, '-', label='{0}: {1:.4E}'.format(f, int(totalview)), color=c)
            ax1.plot_date((start_date, start_date), (0, ax1.get_ylim()[1]), '-', color=c)

            if not totalshare == 'N':
                ax2_y = map(int, dailyshare.split(','))
                ax2_x = [start_date + timedelta(days=i) for i in xrange(len(ax2_y))]
                ax2.plot_date(ax2_x, ax2_y, '-', label='{0}: {1:.4E}'.format(f, int(totalshare)), color=c)
                ax2.plot_date((start_date, start_date), (0, ax2.get_ylim()[1]), '-', color=c)
        cnt += 1

ax1.get_yaxis().set_major_formatter(k_formatter)
ax1.set_ylabel('viewcount')
ax1.set_title('Historical viewcount of top 7 videos of {0}'.format(artist_name))
ax1.legend(loc='best')
ax2.get_yaxis().set_major_formatter(k_formatter)
ax2.set_ylabel('tweetcount')
ax2.set_title('Historical tweetcount of top 7 videos of {0}'.format(artist_name))
ax2.legend(loc='best')

plt.tight_layout()
plt.show()
