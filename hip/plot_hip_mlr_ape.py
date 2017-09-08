#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot absolute percentile error for hip and mlr result, with option to show Honglin's result"""

from __future__ import print_function, division
import sys, os, time, datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from utils.helper import read_as_float_array


def _load_data(filename):
    vid_volume_dict = {}
    with open(os.path.join(data_prefix_dir, '{0}.csv'.format(filename))) as fin:
        fin.readline()
        for line in fin:
            vid, series = line.rstrip().split('\t', 1)
            vid_volume_dict[vid] = np.sum(read_as_float_array(series, delimiter='\t'))
    return vid_volume_dict


def _convert_volume_to_percentile(percentile_map, vid_volume_dict):
    return {vid: stats.percentileofscore(percentile_map, volume) for vid, volume in vid_volume_dict.items()}


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    start_time = time.time()
    sanity_check_honglin = False

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    data_prefix_dir = './data/'
    true_data_list = ['true_view', 'true_watch']
    forecast_data_list = ['hip_view', 'mlr_view', 'mlr_view_share', 'honglin_view', 'honglin_view_share', 'hip_watch', 'mlr_watch', 'mlr_watch_share']
    true_data_dict = {name: _load_data(name) for name in true_data_list}
    forecast_data_dict = {name: _load_data(name) for name in forecast_data_list}
    view_percentile_map = true_data_dict['true_view'].values()
    watch_percentile_map = true_data_dict['true_watch'].values()
    true_view_percentile = _convert_volume_to_percentile(view_percentile_map, true_data_dict['true_view'])
    true_watch_percentile = _convert_volume_to_percentile(watch_percentile_map, true_data_dict['true_watch'])
    print('>>> Finish loading data.')

    # == == == == == == == == Part 3: Convert forecast volume into corresponding percentile == == == == == == == == #
    forecast_percentile_dict = {}
    for name in forecast_data_list:
        if 'view' in name:
            forecast_percentile_dict[name] = _convert_volume_to_percentile(view_percentile_map, forecast_data_dict[name])
        else:
            forecast_percentile_dict[name] = _convert_volume_to_percentile(watch_percentile_map, forecast_data_dict[name])
    print('>>> Finish converting data to percentile.')

    # == == == == == == == == Part 4: Construct target absolute percentile error == == == == == == == == #
    # use an intersect of vids, keys from hip view results
    intersect_vids = forecast_percentile_dict['hip_view'].keys()
    ape_matrix = []
    if sanity_check_honglin:
        target_column = forecast_data_list
    else:
        target_column = ['hip_view', 'mlr_view', 'mlr_view_share', 'hip_watch', 'mlr_watch', 'mlr_watch_share']
    for name in target_column:
        if 'view' in name:
            ape_matrix.append([abs(true_view_percentile[vid] - forecast_percentile_dict[name][vid]) for vid in intersect_vids])
        else:
            ape_matrix.append([abs(true_watch_percentile[vid] - forecast_percentile_dict[name][vid]) for vid in intersect_vids])

    # == == == == == == == == Part 5: Plot APE results == == == == == == == == #
    # Fancy box plot style
    if sanity_check_honglin:
        label_array = ['View\nHistory+Share\nHIP', 'View\nHistory\nMLR', 'View\nHistory+Share\nMLR',
                       'HL-View\nHistory\nMLR', 'HL-View\nHistory+Share\nMLR',
                       'Watch\nHistory+Share\nHIP', 'Watch\nHistory\nMLR', 'Watch\nHistory+Share\nMLR']
        fig = plt.figure(figsize=(11, 5))

        hip_boxes = [ape_matrix[0], [], [], [], [], ape_matrix[5], [], []]
        history_boxes = [[], ape_matrix[1], [], [], [], [], ape_matrix[6], []]
        share_boxes = [[], [], ape_matrix[2], [], [], [], [], ape_matrix[7]]
        honglin_boxes = [[], [], [], ape_matrix[3], ape_matrix[4], [], [], []]
        combined_boxes = [hip_boxes, history_boxes, share_boxes, honglin_boxes]
    else:
        label_array = ['View\nHistory+Share\nHIP', 'View\nHistory\nMLR', 'View\nHistory+Share\nMLR',
                       'Watch\nHistory+Share\nHIP', 'Watch\nHistory\nMLR', 'Watch\nHistory+Share\nMLR']
        fig = plt.figure(figsize=(8, 5))

        hip_boxes = [ape_matrix[0], [], [], ape_matrix[3], [], []]
        history_boxes = [[], ape_matrix[1], [], [], ape_matrix[4], []]
        share_boxes = [[], [], ape_matrix[2], [], [], ape_matrix[5]]
        combined_boxes = [hip_boxes, history_boxes, share_boxes]

    ax1 = fig.add_subplot(111)

    boxplots = [ax1.boxplot(box, labels=label_array, showfliers=False, showmeans=True, widths=0.75) for box in combined_boxes]

    box_colors = ['#6495ed', '#ff6347', '#2e8b57', '#000000']
    n_box = len(combined_boxes)

    for bplot, bcolor in zip(boxplots, box_colors[:n_box]):
        plt.setp(bplot['boxes'], color=bcolor)
        plt.setp(bplot['whiskers'], color=bcolor)
        plt.setp(bplot['caps'], color=bcolor)

    # Now fill the boxes with desired colors
    numBoxes = 2 * n_box
    medians = list(range(numBoxes))

    for ii, bplot, bcolor in zip([[0, numBoxes-3], [1, numBoxes-2], [2, numBoxes-1], [3, 4]][:n_box], boxplots, box_colors[:n_box]):
        for i in ii:
            box = bplot['boxes'][i]
            boxX = []
            boxY = []
            for j in range(5):
                boxX.append(box.get_xdata()[j])
                boxY.append(box.get_ydata()[j])
            boxCoords = list(zip(boxX, boxY))
            boxPolygon = Polygon(boxCoords, facecolor=bcolor if i >= numBoxes-3 else 'w')
            ax1.add_patch(boxPolygon)
            # Now draw the median lines back over what we just filled in
            med = bplot['medians'][i]
            medianX = []
            medianY = []
            for j in range(2):
                medianX.append(med.get_xdata()[j])
                medianY.append(med.get_ydata()[j])
                plt.plot(medianX, medianY, color=bcolor if i <= numBoxes-4 else 'w', lw=1.5, zorder=30)
                medians[i] = medianY[0]
            # Finally, overplot the sample averages, with horizontal alignment in the center of each box
            plt.plot([np.average(med.get_xdata())], [np.average(ape_matrix[i])],
                     color=bcolor if i <= numBoxes-4 else 'w',
                     marker='s', markeredgecolor=bcolor if i <= numBoxes-4 else 'w',
                     zorder=30)

    means = [np.mean(x) for x in ape_matrix]
    means_labels = ['{0:.2f}%'.format(s) for s in means]
    pos = range(len(means))
    for tick, label in zip(pos, ax1.get_xticklabels()):
        ax1.text(pos[tick]+1, means[tick]+0.6, means_labels[tick], horizontalalignment='center', size=16, color='k')
    ax1.set_ylabel('absolute percentile error', fontsize=16)
    ax1.tick_params(axis='y', which='major', labelsize=16)
    ax1.tick_params(axis='x', which='major', labelsize=11)

    # remove upper and right edges
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])

    plt.tight_layout(rect=[0.03, 0, 1, 1])
    plt.show()
