import os
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats


if __name__ == '__main__':
    file_loc = '../../data/vevo_channel_statistics'
    artist_viewership = []

    for subdir, _, files in os.walk(file_loc):
        for f in files:
            filepath = os.path.join(subdir, f)
            with open(filepath, 'r') as filedata:
                res = json.loads(filedata.readline())
                viewcount = int(res['statistics']['viewCount'])
                if not viewcount == 0:
                    artist_viewership.append(viewcount)

    fig, ax1 = plt.subplots(1, 1)

    artist_viewership_dict = defaultdict(int)
    for viewcount in artist_viewership:
        artist_viewership_dict[viewcount] += 1

    x_axis = artist_viewership_dict.keys()
    y_axis = [artist_viewership_dict[k] for k in x_axis]

    bins_log10 = np.logspace(np.log10(min(artist_viewership)), np.log10(max(artist_viewership)), 100)
    counts, bin_edges, ignored = ax1.hist(artist_viewership, bins=bins_log10, label='histogram')

    bins_log_len = np.zeros(bins_log10.size)
    for ii in range(counts.size):
        bins_log_len[ii] = bin_edges[ii + 1] - bin_edges[ii]

    shape, loc, scale = scipy.stats.lognorm.fit(artist_viewership, floc=0)
    estimated_mu = np.log10(scale)
    estimated_sigma = shape

    samples_fit_log = scipy.stats.lognorm.pdf(bins_log10, shape, loc=loc, scale=scale)
    ax1.plot(bins_log10, np.multiply(samples_fit_log, bins_log_len) * sum(counts), linewidth=2, label='fitted distribution')

    print estimated_mu
    print estimated_sigma

    ax1.text(5, 0.9 * ax1.get_ylim()[1], 'estimated mu={0:.2f}\nestimated sigma={1:.2f}'.format(estimated_mu, estimated_sigma))

    print min(artist_viewership)
    print np.percentile(artist_viewership, 25)
    print np.median(artist_viewership)
    print np.percentile(artist_viewership, 75)
    print max(artist_viewership)

    # ax1.scatter(x_axis, y_axis, marker='x', s=5, c='b')

    median_value = np.median(artist_viewership)
    mean_value = np.mean(artist_viewership)

    # ax1.axvline(x=median_value)
    # ax1.text(median_value+0.2, 0.8*ax1.get_ylim()[1], 'Median={0:.2f}'.format(median_value))
    #
    # ax1.axvline(x=mean_value)
    # ax1.text(mean_value+1, 0.7*ax1.get_ylim()[1], 'Mean={0:.2f}'.format(mean_value))

    ax1.set_xscale('log')
    # ax1.set_yscale('log')
    ax1.set_xlim(xmin=1)
    ax1.set_ylim(ymin=1)
    ax1.set_xlabel('Viewcount')
    ax1.set_ylabel('Number of VEVO artists')
    ax1.set_title('Figure 1: Viewcount per VEVO artist')
    ax1.legend(loc='best')

    plt.show()
