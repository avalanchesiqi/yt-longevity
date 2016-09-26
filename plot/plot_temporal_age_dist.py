#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = '../../data/full_jun_data/'


def load_age_frequency(filepath):
    longlived = open(filepath.rsplit('/', 1)[0]+'longlived.txt', 'w')
    with open(filepath, 'r') as filedata:
        age_freq_dict = defaultdict(lambda: (0, 0))
        for line in filedata:
            vid, tc, age = line.rstrip().split()
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


if __name__ == '__main__':
    four_ax_age, four_ax_freq_uw, four_ax_freq_w = load_age_frequency(BASE_DIR+'2014/vid_tweetcount_age.txt')
    five_ax_age, five_ax_freq_uw, five_ax_freq_w = load_age_frequency(BASE_DIR+'2015/vid_tweetcount_age.txt')
    six_ax_age, six_ax_freq_uw, six_ax_freq_w = load_age_frequency(BASE_DIR+'2016/vid_tweetcount_age.txt')

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(20, 16)

    # ax1: age-frequency comparison
    ax1.plot(four_ax_age, four_ax_freq_uw, color='red', label="Jun' 2014 data")
    ax1.plot(five_ax_age, five_ax_freq_uw, color='green', label="Jun' 2015 data")
    ax1.plot(six_ax_age, six_ax_freq_uw, color='blue', label="Jun' 2016 data")
    ax1.set_yscale('log')
    ax1.set_xlim(xmin=-100)
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Age-Frequency distribution comparison')
    ax1.legend(loc='best')

    # ax2: age-tweetcount comparison
    ax2.plot(four_ax_age, four_ax_freq_w, color='red', label="Jun' 2014 data")
    ax2.plot(five_ax_age, five_ax_freq_w, color='green', label="Jun' 2015 data")
    ax2.plot(six_ax_age, six_ax_freq_w, color='blue', label="Jun' 2016 data")
    ax2.set_yscale('log')
    ax2.set_xlim(xmin=-100)
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Tweetcount')
    ax2.set_title('Age-Tweetcount distribution comparison')
    ax2.legend(loc='best')

    # ax3: age-frequency comparison, aligned time offset
    ax3.plot(four_ax_age+731, four_ax_freq_uw, color='red', label="Jun' 2014 data")
    ax3.plot(five_ax_age+366, five_ax_freq_uw, color='green', label="Jun' 2015 data")
    ax3.plot(six_ax_age, six_ax_freq_uw, color='blue', label="Jun' 2016 data")
    ax3.set_yscale('log')
    ax3.set_xlim(xmin=-100)
    # ax3.plot((0, 0), (0, ax3.get_ylim()[1]), 'k-')
    # ax3.plot((731, 731), (0, ax3.get_ylim()[1]), 'k-')
    # ax3.plot((366, 366), (0, ax3.get_ylim()[1]), 'k-')
    ax3.set_xlabel('Age')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Age-Frequency distribution comparison')
    ax3.legend(loc='best')

    # ax4: age-tweetcount comparison, aligned time offset
    ax4.plot(four_ax_age+731, four_ax_freq_w, color='red', label="Jun' 2014 data")
    ax4.plot(five_ax_age+366, five_ax_freq_w, color='green', label="Jun' 2015 data")
    ax4.plot(six_ax_age, six_ax_freq_w, color='blue', label="Jun' 2016 data")
    ax4.set_yscale('log')
    ax4.set_xlim(xmin=-100)
    # ax4.plot((0, 0), (0, ax4.get_ylim()[1]), 'k-')
    # ax4.plot((731, 731), (0, ax4.get_ylim()[1]), 'k-')
    # ax4.plot((366, 366), (0, ax4.get_ylim()[1]), 'k-')
    ax4.set_xlabel('Age')
    ax4.set_ylabel('Tweetcount')
    ax4.set_title('Age-Tweetcount distribution comparison')
    ax4.legend(loc='best')

    fig.savefig('jun_data_with_3_years_apart.png', dpi=fig.dpi)
    # plt.show()
