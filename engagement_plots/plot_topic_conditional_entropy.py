#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""calculate conditional entropy between topic type and relative engagement, i.e, I(book; eta),
by constructing 2x20 oc-occurrence matrix
X                0     1
Y     0-0.05    10    15
   0.05-0.10    20    25
       ....
   0.95-1.00    30    35

I(X;Y) = sum(P(x, y) * log( P(x, y)/P(x)/P(y) ))

Time: ~3M
"""

from __future__ import division, print_function
import os, sys, time, datetime, operator
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter


def safe_log2(x):
    if x == 0:
        return 0
    else:
        return np.log2(x)


def get_conditional_entropy(topic_eta):
    # calculate condition entropy when topic appears
    binned_topic_eta = {i: 0 for i in range(bin_num)}
    for eta in topic_eta:
        binned_topic_eta[min(int(eta / bin_gap), bin_num - 1)] += 1

    p_Y_given_x1 = [binned_topic_eta[i] / len(topic_eta) for i in range(bin_num)]
    return -np.sum([p * safe_log2(p) for p in p_Y_given_x1]), [binned_topic_eta[x] for x in sorted(binned_topic_eta.keys())]


def _load_data(filepath):
    with open(filepath, 'r') as fin:
        fin.readline()
        for line in fin:
            _, _, _, _, _, _, _, topics, _, _, _, re30, _ = line.rstrip().split('\t', 12)
            if topics != '' and topics != 'NA':
                topics = topics.split(',')
                re30 = float(re30)
                for topic in topics:
                    if topic in mid_type_dict:
                        freebase_types = mid_type_dict[topic].split(',')
                        for ft in freebase_types:
                            if ft != 'common' and ft != 'type_ontology' and ft != 'type':
                                type_eta_dict[ft].append(re30)
                                type_eta_counter_dict[ft] += 1
    print('>>> Finish loading file {0}!'.format(filepath))
    return


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    start_time = time.time()
    bin_gap = 0.05
    bin_num = int(1 / bin_gap)

    # == == == == == == == == Part 2: Load Freebase dictionary == == == == == == == == #
    freebase_loc = '../../production_data/freebase_mid_type_name.txt'
    mid_type_dict = {}
    with open(freebase_loc, 'r') as fin:
        for line in fin:
            mid, type, _ = line.rstrip().split('\t', 2)
            mid_type_dict[mid] = type

    # == == == == == == == == Part 3: Load dataset == == == == == == == == #
    data_loc = '../../production_data/new_tweeted_dataset_norm/'
    type_eta_dict = defaultdict(list)
    type_eta_counter_dict = defaultdict(int)
    print('>>> Start to load all tweeted dataset...')
    for subdir, _, files in os.walk(data_loc):
        for f in files:
            _load_data(os.path.join(subdir, f))
    print('>>> Finish loading all data!')
    print('>> Number of topic types: {0}\n'.format(len(type_eta_dict)))

    # == == == == == Part 4: Calculate conditional entropy for topic type and relative engagement == == == == == #
    sorted_type_eta_counter = sorted(type_eta_counter_dict.items(), key=operator.itemgetter(1), reverse=True)[:500]
    print('largest 500 clusters')
    print(sorted_type_eta_counter)

    type_conditional_entropy_dict = {}
    for type, _ in sorted_type_eta_counter:
        # type size, conditional entropy, mean eta value
        type_conditional_entropy_dict[type] = (type_eta_counter_dict[type], get_conditional_entropy(type_eta_dict[type])[0], np.mean(type_eta_dict[type]))

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])

    # == == == == == == == == Part 5: Generate bar plots == == == == == == == == #
    to_plot = True
    if to_plot:
        fig = plt.figure(figsize=(8, 6))
        cornflower_blue = (0.3921, 0.5843, 0.9294)
        tomato = (1.0, 0.3882, 0.2784)

        def make_colormap(seq):
            """Return a LinearSegmentedColormap
            seq: a sequence of floats and RGB-tuples. The floats should be increasing
            and in the interval (0,1).
            """
            seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
            cdict = {'red': [], 'green': [], 'blue': []}
            for i, item in enumerate(seq):
                if isinstance(item, float):
                    r1, g1, b1 = seq[i - 1]
                    r2, g2, b2 = seq[i + 1]
                    cdict['red'].append([item, r1, r2])
                    cdict['green'].append([item, g1, g2])
                    cdict['blue'].append([item, b1, b2])
            return mcolors.LinearSegmentedColormap('CustomMap', cdict)
        c = mcolors.ColorConverter().to_rgb
        rvb = make_colormap([cornflower_blue, c('white'), 0.5, c('white'), tomato])

        keys = type_conditional_entropy_dict.keys()
        x_axis = [type_conditional_entropy_dict[x][0] for x in keys]
        y_axis = [type_conditional_entropy_dict[x][1] for x in keys]
        colors = [type_conditional_entropy_dict[x][2] for x in keys]
        plt.scatter(x_axis, y_axis, c=colors, edgecolors='none', cmap=rvb)

        # plot 1st and 10th most determined category
        top10_index = np.argsort(y_axis)[:10]
        for idx in [top10_index[0], top10_index[-1]]:
            if keys[idx] == 'obamabase':
                keys[idx] = 'obama'
            plt.text(x_axis[idx], y_axis[idx], keys[idx], size=16, ha='center', va='bottom')

        # plot uninformative topic, puffinnpolitics
        plt.text(type_conditional_entropy_dict['baseball'][0], type_conditional_entropy_dict['baseball'][1], 'baseball', size=16, ha='center', va='top')
        plt.text(type_conditional_entropy_dict['book'][0], type_conditional_entropy_dict['book'][1], 'book', size=16, ha='center', va='top')
        plt.text(type_conditional_entropy_dict['film'][0], type_conditional_entropy_dict['film'][1], 'film', size=16, ha='center', va='top')

        plt.xscale('log')
        plt.ylim(ymax=-np.sum([p * safe_log2(p) for p in [bin_gap]*bin_num]))
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.xlabel('topic size', fontsize=16)
        plt.ylabel('conditional entropy', fontsize=16)

        cb = plt.colorbar()
        cb.set_label(label='relative engagement $\eta$', size=16)
        cb.ax.tick_params(labelsize=14)

        # inset subfigure
        ax2 = fig.add_axes([0.38, 0.2, 0.4, 0.4])
        width = 1 / 2
        ind = np.arange(20)

        count_freq1 = get_conditional_entropy(type_eta_dict['handcraft'])[1]
        prob1 = [x / np.sum(count_freq1) for x in count_freq1]
        ax2.bar(ind + width * 1 / 2, prob1, width, color=cb.to_rgba(type_conditional_entropy_dict['handcraft'][2]), label='handcraft')

        count_freq2 = get_conditional_entropy(type_eta_dict['obamabase'])[1]
        prob2 = [x / np.sum(count_freq2) for x in count_freq2]
        ax2.bar(ind + width * 3 / 2, prob2, width, color=cb.to_rgba(type_conditional_entropy_dict['obamabase'][2]), label='obama')

        ax2.set_xlim([0, 20])
        ax2.set_ylim([0, 0.25])
        ax2.set_xticks([0, 4, 8, 12, 16, 20])

        def rescale(x, pos):
            'The two args are the value and tick position'
            return '%1.1f' % (x / 20)

        formatter = FuncFormatter(rescale)

        ax2.xaxis.set_major_formatter(formatter)
        ax2.set_xlabel('relative engagement $\eta$', fontsize=8)
        ax2.set_ylabel('engagement distribution', fontsize=8)
        ax2.tick_params(axis='both', which='major', labelsize=8)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.legend(loc='upper right', fontsize=14, frameon=False)

        plt.tight_layout()
        plt.show()
