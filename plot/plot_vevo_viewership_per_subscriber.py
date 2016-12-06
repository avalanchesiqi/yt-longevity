import os
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


if __name__ == '__main__':
    file_loc = '../../data/vevo_channel_statistics'
    artist_viewership = []
    artist_subscribership = []
    subscriber_view_dict = defaultdict(list)

    for subdir, _, files in os.walk(file_loc):
        for f in files:
            filepath = os.path.join(subdir, f)
            with open(filepath, 'r') as filedata:
                res = json.loads(filedata.readline())
                viewcount = int(res['statistics']['viewCount'])
                subscribercount = int(res['statistics']['subscriberCount'])
                artist_viewership.append(viewcount)
                artist_subscribership.append(subscribercount)
                subscriber_view_dict[subscribercount].append(viewcount)

    fig, ax1 = plt.subplots(1, 1)

    ax1.scatter(artist_subscribership, artist_viewership, marker='x', s=2, c='b')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(xmin=1)
    ax1.set_ylim(ymin=1)
    ax1.set_xlabel('# of subscribers')
    ax1.set_ylabel('# of views')
    ax1.set_title('Figure 1: Number of subscribers and views per VEVO artist\nSpearman correlation={0:.2f}'.format(spearmanr(artist_subscribership, artist_viewership)[0]))
    # ax1.legend(loc='best')

    plt.show()
