#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from scipy.stats import gaussian_kde
sns.set(color_codes=True)
sns.set_style("darkgrid")


# Extract global watch perc ~ duration bivariate from a large collection of YouTube videos, 19M dataset


def read_as_int_array(content, truncated=None, delimiter=None):
    """
    Read input as an int array.
    :param content: string input
    :param truncated: head number of elements extracted
    :param delimiter: delimiter string
    :return: a numpy int array
    """
    if truncated is None:
        return np.array(map(int, content.split(delimiter)), dtype=np.uint32)
    else:
        return np.array(map(int, content.split(delimiter)[:truncated]), dtype=np.uint32)


def read_as_float_array(content, truncated=None, delimiter=None):
    """
    Read input as a float array.
    :param content: string input
    :param truncated: head number of elements extracted
    :param delimiter: delimiter string
    :return: a numpy float array
    """
    if truncated is None:
        return np.array(map(float, content.split(delimiter)), dtype=np.float64)
    else:
        return np.array(map(float, content.split(delimiter)[:truncated]), dtype=np.float64)


def strify(iterable_struct):
    """
    Convert an iterable structure to comma separated string
    :param iterable_struct: an iterable structure
    :return: a string with comma separated
    """
    return ','.join(map(str, iterable_struct))


def get_duration_wp_from_file(filepath, init=False):
    if init:
        res = []
    with open(filepath, 'r') as fin:
        fin.readline()
        for line in fin:
            dump, views, watches = line.rstrip().rsplit(None, 2)
            _, duration, _ = dump.split(None, 2)
            duration = int(duration)
            try:
                views = read_as_float_array(views, delimiter=',', truncated=age)
                watches = read_as_float_array(watches, delimiter=',', truncated=age)
            except:
                continue

            view_num = np.sum(views)
            if view_num == 0:
                continue

            avg_watch_perc = np.sum(watches) * 60 / view_num / duration
            if avg_watch_perc > 1:
                avg_watch_perc = 1

            if init:
                res.append(np.array([duration, avg_watch_perc]))
            else:
                duration_wp_tuple.append(np.array([duration, avg_watch_perc]))

    print('>>> Loading data: {0} done!'.format(filepath))
    if init:
        return res


def convert_to_data_frame(mat):
    return pd.DataFrame(np.array(mat), columns=['video duration', 'watch percentage'])


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    age = 120
    font = {'fontname': 'sans-serif', 'fontsize': 16}
    gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5])
    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(gs[1, 0])

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    # input_path1 = '../../data/production_data/random_info/music.txt'
    # input_path2 = '../../data/production_data/vevo_info.txt'
    # input_path1 = '../../data/production_data/billboard_2016_info.txt'
    # input_path2 = '../../data/production_data/billboard_2016_info.txt'
    # input_path3 = '../../data/production_data/billboard_2016_info.txt'
    # input_paths = [input_path1, input_path2, input_path3]
    input_path1 = '../../data/production_data/random_info/news.txt'
    input_path2 = '../../data/production_data/top_news_info.txt'
    input_paths = [input_path1, input_path2]
    data_matrix = [get_duration_wp_from_file(input_path, init=True) for input_path in input_paths]

    # input_path = '../../data/production_data/random_info'
    # duration_wp_tuple = []
    # for subdir, _, files in os.walk(input_path):
    #     for f in files:
    #         get_duration_wp_from_file(os.path.join(subdir, f), init=False)
    # data_matrix = list()
    # data_matrix.append(duration_wp_tuple)

    data_frames = [convert_to_data_frame(mat) for mat in data_matrix]
    print('>>> Finish loading all data and converting to data frame!')

    df_x = [np.log10(df['video duration']) for df in data_frames]
    df_y = [df['watch percentage'] for df in data_frames]
    # KDE for top marginal
    kde_x = [gaussian_kde(np.log10(df["video duration"])) for df in data_frames]
    # KDE for right marginal
    kde_y = [gaussian_kde(df["watch percentage"]) for df in data_frames]

    xmin, xmax = 1, 5
    ymin, ymax = 0, 1
    x = np.linspace(xmin, xmax, 100)
    y = np.linspace(ymin, ymax, 100)

    sns.kdeplot(df_x[0], df_y[0], cmap='Blues', shade=True, shade_lowest=False, ax=ax, alpha=0.6, n_levels=10, zorder=1)
    sns.kdeplot(df_x[1], df_y[1], cmap='Reds', shade=False, shade_lowest=False, ax=ax, alpha=1, n_levels=6, zorder=2)
    # sns.regplot(df_x[2], df_y[2], color='Black', fit_reg=False, marker='x', ax=ax, scatter_kws={'zorder': 3})
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    ax.set_xlabel('video duration', **font)
    ax.set_ylabel('watch percentage', **font)
    ax.tick_params(axis='both', which='major', labelsize=16)

    ax.xaxis.set_ticklabels(['$\mathregular{10^1}$', '$\mathregular{10^{1.5}}$', '$\mathregular{10^2}$',
                             '$\mathregular{10^{2.5}}$', '$\mathregular{10^3}$', '$\mathregular{10^{3.5}}$',
                             '$\mathregular{10^4}$', '$\mathregular{10^{4.5}}$', '$\mathregular{10^5}$'])

    # ax.xaxis.set_ticklabels(['$\mathregular{10^1.8}$', '$\mathregular{10^2}$', '$\mathregular{10^{2.2}}$',
    #                          '$\mathregular{10^{2.4}}$', '$\mathregular{10^{2.6}}$', '$\mathregular{10^{2.8}}$',
    #                          '$\mathregular{10^3}$'])

    for label in ax.get_xticklabels()[1::2]:
        label.set_visible(False)

    ax.legend([plt.Rectangle((0, 0), 1, 1, fc='Blue', alpha=0.50),
               plt.Rectangle((0, 0), 1, 1, fc='Red', alpha=0.50)],
               # plt.Rectangle((0, 0), 1, 1, fc='Black', alpha=0.50)],
              ['Random News', 'Top News'],
              # ['Random Music', 'VEVO', 'BillBoard'],
              loc='upper right', fontsize=16)

    # Create Y-marginal (right)
    max_xlim = 1.2*max([func(y).max() for func in kde_y])
    axr = plt.subplot(gs[1, 1], xticks=[], yticks=[], frameon=False, xlim=(0, max_xlim), ylim=(ymin, ymax))
    axr.plot(kde_y[0](y), y, color='b')
    axr.plot(kde_y[1](y), y, color='r')
    # axr.plot(kde_y[2](y), y, color='k')

    # Create X-marginal (top)
    max_ylim = 1.2 * max([func(x).max() for func in kde_x])
    axt = plt.subplot(gs[0, 0], xticks=[], yticks=[], frameon=False, xlim=(xmin, xmax), ylim=(0, max_ylim))
    axt.plot(x, kde_x[0](x), color='b')
    axt.plot(x, kde_x[1](x), color='r')
    # axt.plot(x, kde_x[2](x), color='k')

    fig.tight_layout(pad=1)

    # ax1.set_title('Top News')
    # x_axis = list(np.logspace(1, np.log10(np.max(duration_array)), bin_num))
    #
    # # get bin statistics
    # try:
    #     for item in sorted_duration_wp_tuple:
    #         while bin_idx < bin_num and item[0] > x_axis[bin_idx]:
    #             bin_idx += 1
    #         bin_matrix[bin_idx].append(item[1])
    # except:
    #     print(bin_idx, x_axis[bin_idx-1])
    #
    # bin_matrix = [np.array(x) for x in bin_matrix]
    # print('videos in each bin')
    # print([len(x) for x in bin_matrix])
    # y_axis = [np.mean(x) for x in bin_matrix]
    #
    # fout = open('global_parameters_news.txt', 'w')
    # fout.write('{0}\n'.format(strify(x_axis)))
    # fout.write('{0}\n'.format(strify(y_axis)))
    # fout.close()
    #
    # ax1.plot(x_axis, y_axis, 'b.--', label='Random Dataset Mean', zorder=1)
    #
    # ax1.set_ylim(ymin=0)
    # ax1.set_ylim(ymax=1)
    # ax1.set_xlabel('Video duration (sec)', fontsize=20)
    # ax1.set_ylabel('Watch percentage', fontsize=20)
    # ax1.set_xscale('log')
    # ax1.tick_params(axis='both', which='major', labelsize=20)
    #
    # plt.legend(loc='upper right', fontsize=20)
    plt.show()
