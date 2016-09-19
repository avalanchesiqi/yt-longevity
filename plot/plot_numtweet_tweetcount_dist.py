# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle
import numpy as np
from collections import defaultdict
from scipy import stats
from datetime import datetime

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.set_size_inches(30, 14)

full_may_nt = np.array(
    [3901000, 3963000, 3903000, 3900000, 3892000, 3991000, 3943000, 3914000, 3812000, 3919000, 4033000, 4111000,
     3964000, 3982000, 3978000, 3958000, 3937000, 3986000, 3925000, 3893000, 3834000, 3874000, 3887000, 3897000,
     3933000, 3878000, 3972000, 3807000, 3842000, 3847000, 3764000])
sample_may_nt = np.array(
    [390220, 395238, 390173, 389438, 389675, 398356, 395024, 390214, 382316, 392537, 402749, 410056, 395693, 398904,
     398770, 396527, 394629, 398588, 392294, 389232, 384666, 387300, 388685, 389691, 393726, 387486, 396926, 380129,
     384442, 384583, 376474])

date_xaxis = np.array([datetime(2016, 5, i+1) for i in xrange(len(full_may_nt))])

# ax1 frequency ~ number of tweets
ax1.plot_date(date_xaxis, full_may_nt, color='blue', label='full may data')
ax1.plot_date(date_xaxis, sample_may_nt, color='red', label='sample may data')
ax1.xaxis.set_major_locator(DayLocator(interval=5))
ax1.xaxis.set_major_formatter(DateFormatter('%d-%m-%Y'))
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: '{0:.1f}M'.format(1.0*x/1000000)))
ax1.set_ylim(ymin=0)
ax1.set_xlabel('Date')
ax1.set_ylabel('Frequency')
ax1.set_title('Full May vs Sampled May Number of Tweets Frequency Distribution')
ax1.legend(loc='best')

def get_cdf(arr):
    total = np.sum(arr)
    return np.array([1.0*np.sum(arr[:i+1])/total for i in xrange(len(arr))])

# ax2 dcdf ~ number of tweets
nt_sample_may_dcdf = get_cdf(sample_may_nt)
nt_full_may_dcdf = get_cdf(full_may_nt)
nt_s, nt_p = stats.ks_2samp(nt_sample_may_dcdf, nt_full_may_dcdf)

nt_full_may_plot = ax2.plot_date(date_xaxis, nt_full_may_dcdf, alpha=0.5, color='blue', label='full may data')[0]
nt_sample_may_plot = ax2.plot_date(date_xaxis, nt_sample_may_dcdf, alpha=0.5, color='red', label='sample may data')[0]
ax2.xaxis.set_major_locator(DayLocator(interval=5))
ax2.xaxis.set_major_formatter(DateFormatter('%d-%m-%Y'))
ax2.set_ylim(ymin=0)
ax2.set_xlabel('Date')
ax2.set_ylabel('Cumulative Distribution Function')
ax2.set_title('Full May vs Sampled May Number of Tweets Discrete CDF')
extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
ax2.legend([nt_full_may_plot, nt_sample_may_plot, extra], ('full may data', 'sample may data', 'pvalue of ks test: {0:.2E}'.format(nt_p)), loc='best')

# load tweetcount data
def loaddata(filepath):
    datafile = open(filepath, 'r')
    tc_dict = defaultdict(int)
    for line in datafile:
        vid, tc = line.rstrip().split()
        tc_dict[int(tc)] += 1
    datafile.close()

    ax_x = np.array([0], dtype=int)
    ax_freq = np.array([0], dtype=int)
    for k, v in sorted(tc_dict.items()):
        ax_x = np.append(ax_x, k)
        ax_freq = np.append(ax_freq, v)

    # convert to discrete cdf
    total = np.sum(ax_freq)
    ax_dcdf = np.array([1.0*(np.sum(ax_freq[:i+1]))/total for i in xrange(len(ax_freq))])

    print 'Finish loading tweetcount data from {0}.'.format(filepath)
    return ax_x, ax_freq, ax_dcdf

# sample_may_x, sample_may_freq, sample_may_dcdf = loaddata('../../data/sample_may_data/vid_tweetcount.txt')
# # full_may_x, full_may_freq, full_may_dcdf = loaddata('full_data/vid_tweetcount.txt')
#
# # ax3 frequency ~ tweetcount
# # ax3.scatter(full_may_x, full_may_freq, color='blue', s=0.5, label='full may data')
# ax3.scatter(sample_may_x, sample_may_freq, color='red', s=0.5, label='sample may data')
# ax3.set_xscale('symlog')
# ax3.set_yscale('symlog')
# ax3.set_xlim(xmin=0)
# ax3.set_ylim(ymin=0)
# ax3.set_title('Full May vs Sampled May Tweetcount Frequency Distribution')
# ax3.set_xlabel('Tweetcount')
# ax3.set_ylabel('Frequency')
# ax3.legend(loc='best')
#
# # ax4 dcdf ~ tweetcount
# # tc_s, tc_p = stats.ks_2samp(sample_may_dcdf, full_may_dcdf)
#
# # tc_full_may_plot = ax4.plot(full_may_x, full_may_dcdf, color='blue', label='full may data')[0]
# tc_sample_may_plot = ax4.plot(sample_may_x, sample_may_dcdf, color='red', label='sample may data')[0]
# ax4.set_xscale('symlog')
# ax4.set_xlim(xmin=0)
# ax4.set_ylim(ymin=0)
# ax4.set_title('Full May vs Sampled May Tweetcount Discrete CDF')
# ax4.set_xlabel('Tweetcount')
# ax4.set_ylabel('Cumulative Distribution Function')
# # ax4.legend([tc_full_may_plot, tc_sample_may_plot, extra], ('full may data', 'sample may data', 'pvalue of ks test: {0:.4E}'.format(tc_p)), loc='best')

# fig.savefig('figs/full_sample_may_tweets_tweetcount_dist.png', dpi=fig.dpi)

# plt.tight_layout()
plt.show()
