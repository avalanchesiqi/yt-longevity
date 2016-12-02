import os
import json
from collections import defaultdict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    file_loc = '../../data/vevo'
    vevo_dailyview_dict = defaultdict(int)

    for subdir, _, files in os.walk(file_loc):
        for f in files:
            filepath = os.path.join(subdir, f)
            cnt = 0
            with open(filepath, 'r') as filedata:
                for line in filedata:
                    if line.rstrip():
                        video = json.loads(line.rstrip())
                        try:
                            start_date = datetime(*map(int, video['insights']['startDate'].split('-')))
                            dailyviews = map(int, video['insights']['dailyView'].split(','))
                            for i in xrange(len(dailyviews)):
                                vevo_dailyview_dict[start_date+timedelta(days=i)] += dailyviews[i]
                        except:
                            continue

    fig, ax1 = plt.subplots(1, 1)

    x_axis = sorted(vevo_dailyview_dict.keys())
    y_axis = [vevo_dailyview_dict[k] for k in x_axis]

    ax1.plot_date(x_axis, y_axis, 'o-', c='b', ms=3)

    ax1.set_xlabel('Calender date')
    ax1.set_ylabel('Number of VEVO video views')
    ax1.set_title('Figure 1: Daily VEVO video view trend.')

    plt.show()
