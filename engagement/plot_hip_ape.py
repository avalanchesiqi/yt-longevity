from __future__ import print_function, division
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


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


def generate_metric_percentile(arr, bin_num=400):
    res = []
    for i in xrange(1, bin_num+1):
        res.append(np.percentile(arr, i*100/bin_num))
    return np.array(res)


def lookup_percentile(mapping, query):
    return (np.sum(mapping < query) + np.sum(mapping <= query)) * 50 / float(len(mapping))


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    hip_files = ['./data/training_view_reg_tune.log', './data/training_watch_reg_tune.log']
    mlr_files = ['./data/mlr_daily_forecast_view_wo_share.txt', './data/mlr_daily_forecast_watch_wo_share.txt']
    share_files = ['./data/mlr_daily_forecast_view_w_share.txt', './data/mlr_daily_forecast_watch_w_share.txt']
    ape_matrix = []

    for hip_file, mlr_file, share_file in zip(hip_files, mlr_files, share_files):
        mlr_dict = {}
        share_dict2 = {}
        with open(mlr_file, 'r') as mlr_fin:
            for line in mlr_fin:
                vid, predict = line.split(',', 1)
                mlr_dict[vid] = np.sum(read_as_float_array(predict, delimiter=','))
                share_dict2[vid] = read_as_float_array(predict, delimiter=',')
        if 'view' in share_file:
            pickle.dump(share_dict2, open('data/hip/mlr_view_wo_share.p', 'wb'))
        if 'watch' in share_file:
            pickle.dump(share_dict2, open('data/hip/mlr_watch_wo_share.p', 'wb'))

        share_dict = {}
        with open(share_file, 'r') as share_fin:
            for line in share_fin:
                vid, predict = line.split(',', 1)
                share_dict[vid] = np.sum(read_as_float_array(predict, delimiter=','))

        true_dict = {}
        hip_dict = {}
        with open(hip_file, 'r') as hip_fin:
            for line in hip_fin:
                dump, daily_metric, true_total, hip_total = line.rstrip().rsplit(None, 3)
                vid, _ = dump.split(None, 1)
                if vid in mlr_dict:
                    observed_ninety = np.sum(read_as_float_array(daily_metric, delimiter=',', truncated=90))
                    true_dict[vid] = float(true_total) - observed_ninety
                    hip_dict[vid] = float(hip_total) - observed_ninety

        vids = true_dict.keys()
        true_arr = [true_dict[vid] for vid in vids]
        hip_arr = [hip_dict[vid] for vid in vids]
        mlr_arr = [mlr_dict[vid] for vid in vids]
        share_arr = [share_dict[vid] for vid in vids]

        mapping = generate_metric_percentile(true_arr)
        ape_matrix.append([abs(lookup_percentile(mapping, hip_arr[x]) - lookup_percentile(mapping, true_arr[x])) for x in xrange(len(hip_arr))])
        ape_matrix.append([abs(lookup_percentile(mapping, mlr_arr[x]) - lookup_percentile(mapping, true_arr[x])) for x in xrange(len(hip_arr))])
        ape_matrix.append([abs(lookup_percentile(mapping, share_arr[x]) - lookup_percentile(mapping, true_arr[x])) for x in xrange(len(hip_arr))])

    # == == == == == == == == Part 2: Plot APE results == == == == == == == == #
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(111)

    hip_model = [ape_matrix[0], [], [], ape_matrix[3], [], []]
    mlr_model = [[], ape_matrix[1], [], [], ape_matrix[4], []]
    mlr_model2 = [[], [], ape_matrix[2], [], [], ape_matrix[5]]

    label_array = ['View\nHistory+Share\nHIP', 'View\nHistory\nMLR', 'View\nHistory+Share\nMLR', 'Watch\nHistory+Share\nHIP', 'Watch\nHistory\nMLR', 'Watch\nHistory+Share\nMLR']
    bplot1 = ax1.boxplot(hip_model, labels=label_array, showfliers=False, showmeans=True, widths=0.75)
    bplot2 = ax1.boxplot(mlr_model, labels=label_array, showfliers=False, showmeans=True, widths=0.75)
    bplot3 = ax1.boxplot(mlr_model2, labels=label_array, showfliers=False, showmeans=True, widths=0.75)
    ax1.set_ylabel('absolute percentile error', fontsize=16)
    boxColors = ['#6495ed', '#ff6347', '#2e8b57']
    plt.setp(bplot1['boxes'], color=boxColors[0])
    plt.setp(bplot1['whiskers'], color=boxColors[0])
    plt.setp(bplot1['caps'], color=boxColors[0])

    plt.setp(bplot2['boxes'], color=boxColors[1])
    plt.setp(bplot2['whiskers'], color=boxColors[1])
    plt.setp(bplot2['caps'], color=boxColors[1])

    plt.setp(bplot3['boxes'], color=boxColors[2])
    plt.setp(bplot3['whiskers'], color=boxColors[2])
    plt.setp(bplot3['caps'], color=boxColors[2])

    # Now fill the boxes with desired colors

    numBoxes = 2 * 3
    medians = list(range(numBoxes))

    for ii, bplot, c in zip([[0, 3], [1, 4], [2, 5]], [bplot1, bplot2, bplot3], boxColors):
        for i in ii:
            box = bplot['boxes'][i]
            boxX = []
            boxY = []
            for j in range(5):
                boxX.append(box.get_xdata()[j])
                boxY.append(box.get_ydata()[j])
            boxCoords = list(zip(boxX, boxY))
            # Alternate between Dark Khaki and Royal Blue
            k = i % 2
            boxPolygon = Polygon(boxCoords, facecolor=c if i > 2 else 'w')
            ax1.add_patch(boxPolygon)
            # Now draw the median lines back over what we just filled in
            med = bplot['medians'][i]
            medianX = []
            medianY = []
            for j in range(2):
                medianX.append(med.get_xdata()[j])
                medianY.append(med.get_ydata()[j])
                plt.plot(medianX, medianY, color=c if i < 3 else 'w', lw=1.5, zorder=30)
                medians[i] = medianY[0]
            # Finally, overplot the sample averages, with horizontal alignment
            # in the center of each box
            plt.plot([np.average(med.get_xdata())], [np.average(ape_matrix[i])],
                     color=c if i < 3 else 'w', marker='s', markeredgecolor=c if i < 3 else 'w', zorder=30)

    means = [np.mean(x) for x in ape_matrix]
    means_labels = ['{0:.2f}%'.format(s) for s in means]
    pos = range(len(means))
    for tick, label in zip(pos, ax1.get_xticklabels()):
        ax1.text(pos[tick]+1, means[tick]+0.6, means_labels[tick], horizontalalignment='center', size=16, color='k')
    ax1.tick_params(axis='y', which='major', labelsize=16)
    ax1.tick_params(axis='x', which='major', labelsize=11)

    # remove upper and right edges
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')

    plt.tight_layout()
    plt.show()
