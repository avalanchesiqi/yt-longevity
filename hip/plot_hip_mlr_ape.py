from __future__ import print_function, division
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error
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


def lookup_percentile(mapping, query):
    return stats.percentileofscore(mapping, query)


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    prefix_dir = './output/'
    hip_files = [prefix_dir+'training_view_reg_tune.log', prefix_dir+'training_watch_reg_tune.log']
    mlr_history_files = [prefix_dir+'mlr_forecast_view_without_share.txt', prefix_dir+'mlr_forecast_watch_without_share.txt']
    mlr_share_files = [prefix_dir+'mlr_forecast_view_with_share.txt', prefix_dir+'mlr_forecast_watch_with_share.txt']
    ape_matrix = []

    for hip_file, history_file, share_file in zip(hip_files, mlr_history_files, mlr_share_files):
        true_dict = {}
        true_verbose_dict = {}
        hip_dict = {}
        with open(hip_file, 'r') as hip_fin:
            for line in hip_fin:
                dump, daily_metric, true_total, hip_total = line.rstrip().rsplit(None, 3)
                vid, _ = dump.split(None, 1)
                observed_ninety = np.sum(read_as_float_array(daily_metric, delimiter=',', truncated=90))
                true_verbose_dict[vid] = read_as_float_array(daily_metric, delimiter=',')
                true_dict[vid] = float(true_total) - observed_ninety
                hip_dict[vid] = float(hip_total) - observed_ninety

        history_dict = {}
        history_verbose_dict = {}
        history_norm_mse_dict = {}
        with open(history_file, 'r') as mlr_fin:
            for line in mlr_fin:
                vid, predict = line.split(',', 1)
                if vid in true_dict.keys():
                    history_dict[vid] = np.sum(read_as_float_array(predict, delimiter=','))
                    history_verbose_dict[vid] = read_as_float_array(predict, delimiter=',')
                    history_norm_mse_dict[vid] = np.sum([((np.sum(true_verbose_dict[vid][:90])+np.sum(history_verbose_dict[vid][:i-90]))/np.sum(true_verbose_dict[vid][:i]) - 1)**2 for i in range(91, 120)])

        share_dict = {}
        share_verbose_dict = {}
        share_norm_mse_dict = {}
        with open(share_file, 'r') as share_fin:
            for line in share_fin:
                vid, predict = line.split(',', 1)
                if vid in true_dict.keys():
                    share_dict[vid] = np.sum(read_as_float_array(predict, delimiter=','))
                    share_verbose_dict[vid] = read_as_float_array(predict, delimiter=',')
                    share_norm_mse_dict[vid] = np.sum([((np.sum(true_verbose_dict[vid][:90])+np.sum(share_verbose_dict[vid][:i-90]))/np.sum(true_verbose_dict[vid][:i]) - 1) ** 2 for i in range(91, 120)])

        percentile_map = np.array([true_dict[vid] for vid in history_dict.keys()])
        true_percentile = [lookup_percentile(percentile_map, true_dict[vid]) for vid in history_dict.keys()]
        hip_percentile = [lookup_percentile(percentile_map, hip_dict[vid]) for vid in history_dict.keys()]
        history_percentile = [lookup_percentile(percentile_map, history_dict[vid]) for vid in history_dict.keys()]
        share_percentile = [lookup_percentile(percentile_map, share_dict[vid]) for vid in history_dict.keys()]

        ape_matrix.append([abs(x-y) for x, y in zip(true_percentile, hip_percentile)])
        ape_matrix.append([abs(x-y) for x, y in zip(true_percentile, history_percentile)])
        ape_matrix.append([abs(x-y) for x, y in zip(true_percentile, share_percentile)])

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

    plt.tight_layout(rect=[0.03, 0, 1, 1])
    plt.show()
