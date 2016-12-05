import os
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


def plot_dist(artist_arr, ax):
    bins_log10 = np.logspace(np.log10(min(artist_arr)), np.log10(max(artist_arr)), 100)
    counts, bin_edges, ignored = ax.hist(artist_arr, bins=bins_log10, label='histogram')

    bins_log_len = np.zeros(bins_log10.size)
    for ii in range(counts.size):
        bins_log_len[ii] = bin_edges[ii + 1] - bin_edges[ii]

    shape, loc, scale = scipy.stats.lognorm.fit(artist_arr, floc=0)
    estimated_mu = np.log10(scale)
    estimated_sigma = shape

    samples_fit_log = scipy.stats.lognorm.pdf(bins_log10, shape, loc=loc, scale=scale)
    ax.plot(bins_log10, np.multiply(samples_fit_log, bins_log_len) * sum(counts), linewidth=2,
            label='fitted distribution')

    ax.text(3, 0.9 * ax.get_ylim()[1],
            'estimated mu={0:.2f}\nestimated sigma={1:.2f}'.format(estimated_mu, estimated_sigma))

    print 'estimated mu', estimated_mu
    print 'estimated sigma', estimated_sigma
    print 'mean', np.log10(np.mean(artist_arr))
    print 'std', np.log10(np.std(artist_arr))
    print 'min', min(artist_arr)
    print '25q', np.percentile(artist_arr, 25)
    print 'median', np.median(artist_arr)
    print '75q', np.percentile(artist_arr, 75)
    print 'max', max(artist_arr)
    print '---------------------------'

    ax.set_xscale('log')
    ax.set_xlim(xmin=1)
    ax.set_ylim(ymin=1)
    ax.legend(loc='best')


if __name__ == '__main__':
    file_loc = '../../data/vevo_channel_statistics'
    artist_viewership = []
    artist_subscribership = []

    for subdir, _, files in os.walk(file_loc):
        for f in files:
            filepath = os.path.join(subdir, f)
            with open(filepath, 'r') as filedata:
                res = json.loads(filedata.readline())
                viewcount = int(res['statistics']['viewCount'])
                subscribercount = int(res['statistics']['subscriberCount'])
                if not viewcount == 0:
                    artist_viewership.append(viewcount)
                if not subscribercount == 0:
                    artist_subscribership.append(subscribercount)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    plot_dist(artist_viewership, ax1)
    plot_dist(artist_subscribership, ax2)

    ax1.set_xlabel('Viewership')
    ax1.set_ylabel('Number of VEVO artists')
    ax1.set_title('Figure 1: Viewership per VEVO artist')

    ax2.set_xlabel('Subscriber Count')
    ax2.set_ylabel('Number of VEVO artists')
    ax2.set_title('Figure 2: Subscriber per VEVO artist')

    plt.show()
