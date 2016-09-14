#!/usr/bin/env python

import os
import json
from collections import defaultdict
from datetime import datetime
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


BASE_DIR = '../../data/'


def write_vid_age_tweetcount(dir_path):
    vid_publish = {}
    for subdir, _, files in os.walk(dir_path+'/video_metadatas'):
        for f in files:
            if f.endswith('json'):
                filepath = os.path.join(subdir, f)
                with open(filepath, 'r') as datefile:
                    for line in datefile:
                        metadata = json.loads(line.rstrip())
                        try:
                            vid_publish[metadata['id']] = metadata['snippet']['publishedAt'][:10]
                        except Exception as exc:
                            print metadata['id'], str(exc)
                            continue
                print f.title(), "is loaded!"

    with open(dir_path+'/vid_age_tweetcount.txt', 'w') as output_file:
        for subdir, _, files in os.walk(dir_path+'/video_stats'):
            for f in sorted(files):
                tweet_date = f[:10]
                print 'now processing tweet date', f[:10]
                filepath = os.path.join(subdir, f)
                with open(filepath, 'r') as datefile:
                    for line in datefile:
                        vid, tc = line.rstrip().split()
                        if vid in vid_publish:
                            upload_date = vid_publish[vid]
                            age = (datetime(*map(int, tweet_date.split('-'))) - datetime(*map(int, upload_date.split('-')))).days
                            output_file.write('{0}\t{1}\t{2}\n'.format(vid, age, tc))


def load_age_frequency(filepath):
    longlived = open('../datasets/'+filepath.split('/')[-2]+'_longlived.txt', 'w')
    with open(filepath, 'r') as datafile:
        age_freq_dict = defaultdict(lambda: (0, 0))
        for line in datafile:
            vid, age, tc = line.rstrip().split()
            if int(age) >= 0:
                unweighted, weighted = age_freq_dict[int(age)]
                age_freq_dict[int(age)] = (unweighted+1, weighted+int(tc))
                if int(age) >= 500 and int(tc) >= 50:
                    longlived.write(line)
    longlived.close()

    ax_age = np.array([], dtype=int)
    ax_freq_uw = np.array([], dtype=int)
    ax_freq_w = np.array([], dtype=int)
    for age, freq in sorted(age_freq_dict.items()):
        ax_age = np.append(ax_age, age)
        ax_freq_uw = np.append(ax_freq_uw, freq[0])
        ax_freq_w = np.append(ax_freq_w, freq[1])

    print 'Finish loading age-frequency from {0}.'.format(filepath)
    return ax_age, ax_freq_uw, ax_freq_w


def load_tweetcount_frequency(filepath):
    with open(filepath, 'r') as datafile:
        tc_freq_dict = defaultdict(int)
        for line in datafile:
            vid, age, tc = line.rstrip().split()
            if int(tc) >= 0:
                tc_freq_dict[int(tc)] += 1

    ax_tc = np.array([], dtype=int)
    ax_freq = np.array([], dtype=int)
    for tc, freq in sorted(tc_freq_dict.items()):
        ax_tc = np.append(ax_tc, tc)
        ax_freq = np.append(ax_freq, freq)

    print 'Finish loading tweetcount-frequency from {0}.'.format(filepath)
    return ax_tc, ax_freq


if __name__ == '__main__':
    # write_vid_age_tweetcount(BASE_DIR+'sample_may_2016_data')
    # write_vid_age_tweetcount(BASE_DIR+'sample_jun_2014_data')

    may_ax_age, may_ax_freq_uw, may_ax_freq_w = load_age_frequency(BASE_DIR+'sample_may_2016_data/vid_age_tweetcount.txt')
    jun_ax_age, jun_ax_freq_uw, jun_ax_freq_w = load_age_frequency(BASE_DIR+'sample_jun_2014_data/vid_age_tweetcount.txt')

    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # ax1: sample may-2016 / sample jun-2014 age-frequency comparison
    ax1.plot(may_ax_age, may_ax_freq_uw, color='blue', label='sample may-2016 data')
    ax1.plot(jun_ax_age, jun_ax_freq_uw, color='red', label='sample jun-2014 data')
    ax1.plot(jun_ax_age+700, jun_ax_freq_uw, color='green', label='sample jun-2014 data, aligned time offset')
    ax1.set_yscale('log')
    ax1.set_xlim(xmin=-100)
    ax1.plot((0, 0), (0, ax1.get_ylim()[1]), 'k-')
    ax1.plot((700, 700), (0, ax1.get_ylim()[1]), 'k-')
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Age-Frequency distribution comparison')
    ax1.legend(loc='best')

    # ax2: sample may-2016 / sample jun-2014 age-frequency comparison, aligned time offset
    ax2.plot(may_ax_age, may_ax_freq_w, color='blue', label='sample may-2016 data')
    ax2.plot(jun_ax_age, jun_ax_freq_w, color='red', label='sample jun-2014 data')
    ax2.plot(jun_ax_age+700, jun_ax_freq_w, color='green', label='sample jun-2014 data, aligned time offset')
    ax2.set_yscale('log')
    ax2.set_xlim(xmin=-100)
    ax2.plot((0, 0), (0, ax2.get_ylim()[1]), 'k-')
    ax2.plot((700, 700), (0, ax2.get_ylim()[1]), 'k-')
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Tweetcount')
    ax2.set_title('Age-Tweetcount distribution comparison')
    ax2.legend(loc='best')

    may_ax_tc, may_ax_freq2 = load_tweetcount_frequency(BASE_DIR + 'sample_may_2016_data/vid_age_tweetcount.txt')
    jun_ax_tc, jun_ax_freq2 = load_tweetcount_frequency(BASE_DIR + 'sample_jun_2014_data/vid_age_tweetcount.txt')

    # # ax3: sample may-2016 / sample jun-2014 tweetcount-frequency comparison
    # ax3.scatter(may_ax_tc, may_ax_freq2, color='blue', s=0.5, label='sample may-2016 data')
    # ax3.scatter(jun_ax_tc, jun_ax_freq2, color='red', s=0.5, label='sample jun-2014 data')
    # ax3.set_xscale('symlog')
    # ax3.set_yscale('symlog')
    # ax3.set_xlim(xmin=0)
    # ax3.set_ylim(ymin=0)
    # ax3.set_xlabel('Tweetcount')
    # ax3.set_ylabel('Frequency')
    # ax3.set_title('Tweetcount-Frequency distribution comparison')
    # ax3.legend(loc='best')
    #
    # ax4.plot(may_ax_age, 1.0*may_ax_freq_w/may_ax_freq_uw, color='blue', label='sample may-2016 data')
    # ax4.plot(jun_ax_age, 1.0*jun_ax_freq_w/jun_ax_freq_uw, color='red', label='sample jun-2014 data')
    # ax4.set_xlim(xmin=-100)
    # ax4.set_xlabel('Age')
    # ax4.set_ylabel('Frequency')
    # ax4.set_title('Age-Frequency distribution comparison')
    # ax4.legend(loc='best')

    # # ax4: sample may-2016 / sample jun-2014 tweetcount-dcdf comparison
    # # convert to discrete cdf
    # may_ax_dcdf2 = np.array([1.0*(np.sum(may_ax_freq2[:i+1]))/np.sum(may_ax_freq2) for i in xrange(len(may_ax_freq2))])
    # jun_ax_dcdf2 = np.array([1.0*(np.sum(jun_ax_freq2[:i+1]))/np.sum(jun_ax_freq2) for i in xrange(len(jun_ax_freq2))])
    # may_ax_tc = np.append([0], may_ax_tc)
    # jun_ax_tc = np.append([0], jun_ax_tc)
    # may_ax_dcdf2 = np.append([0], may_ax_dcdf2)
    # jun_ax_dcdf2 = np.append([0], jun_ax_dcdf2)
    #
    # tc_s, tc_p = stats.ks_2samp(may_ax_dcdf2, jun_ax_dcdf2)
    #
    # tc_sample_may_plot = ax4.plot(may_ax_tc, may_ax_dcdf2, color='blue')[0]
    # tc_sample_jun_plot = ax4.plot(jun_ax_tc, jun_ax_dcdf2, color='red')[0]
    # ax4.set_xscale('symlog')
    # ax4.set_xlim(xmin=0)
    # ax4.set_ylim(ymin=0)
    # ax4.set_title('Tweetcount-Discrete CDF distribution comparison')
    # ax4.set_xlabel('Tweetcount')
    # ax4.set_ylabel('Cumulative Distribution Function')
    # extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    # ax4.legend([tc_sample_may_plot, tc_sample_jun_plot, extra], ('sample may-2016 data', 'sample jun-2014 data', 'pvalue of ks test: {0:.4E}'.format(tc_p)), loc='best')

    # fig.savefig('../figs/comp_sample_may_jun_age_dist.png', dpi=fig.dpi)
    plt.show()
