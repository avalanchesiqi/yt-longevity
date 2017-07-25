from __future__ import print_function, division
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


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


def get_mean_prec(bin_idx):
    return mean_watch_prec[bin_idx]


def get_median_prec(bin_idx):
    return median_watch_prec[bin_idx]


def get_upper_bound(bin_idx):
    return upper_watch_prec[bin_idx]


def get_lower_bound(bin_idx):
    return lower_watch_prec[bin_idx]


def plot_performance_comp(matrix, fig_idx):
    labels = ['D', 'D+C', 'D+T', 'D+C+T', 'D+U', 'D+U+C', 'D+U+T', 'D+U+C+T']
    ax = fig.add_subplot(111+fig_idx)
    print(len(matrix))
    ax.boxplot(matrix, labels=labels[:len(matrix)], showfliers=False, showmeans=True)
    ax.set_ylabel('absolute percentile error')

    means = [np.mean(x) for x in matrix]
    means_labels = ['{0:.2f}%'.format(s) for s in means]
    pos = range(len(means))
    for tick, label in zip(pos, ax.get_xticklabels()):
        ax.text(pos[tick] + 1, means[tick] + 0.1, means_labels[tick], horizontalalignment='center', size='medium',
                color='k')


def plot_r2_comp(matrix, fig_idx):
    labels = ['D', 'D+C', 'D+T', 'D+C+T', 'D+U', 'D+U+C', 'D+U+T', 'D+U+C+T']
    ax = fig.add_subplot(211 + fig_idx)
    x_pos = np.arange(len(matrix))

    ax.bar(x_pos, matrix, align='center')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels[:len(matrix)])

    ax.set_ylabel('r2 score')

    values_labels = ['{0:.3f}'.format(s) for s in matrix]
    for tick, label in zip(x_pos, ax.get_xticklabels()):
        ax.text(x_pos[tick], matrix[tick] + 0.05, values_labels[tick], horizontalalignment='center', size='medium',
                color='k')


def plot_dur_wp_dist(dur_wp_tuple, fig_idx, x_label):
    bin_matrix = []
    bin_list = []
    bin_idx = 0
    sorted_tuple = sorted(dur_wp_tuple, key=lambda x: x[0])
    # put dur-wp tuple in the correct bin
    for item in sorted_tuple:
        if item[0] > duration_gap[bin_idx]:
            # x_axis.append(duration_gap[bin_idx])
            bin_matrix.append(bin_list)
            bin_idx += 1
            bin_list = []
        bin_list.append(item[1])
    if len(bin_list) > 0:
        bin_matrix.append(bin_list)
    bin_matrix = [np.array(x) for x in bin_matrix]

    ax = fig.add_subplot(330+fig_idx)
    ax.plot(duration_gap, [np.percentile(x, 5) for x in bin_matrix], 'g--', label='5%', zorder=1)
    ax.plot(duration_gap, [np.mean(x) for x in bin_matrix], 'r-', label='Mean', zorder=1)
    ax.plot(duration_gap, [np.percentile(x, 95) for x in bin_matrix], 'g--', label='95%', zorder=1)

    ax.set_xlim(xmin=duration_gap[0])
    ax.set_ylim([0, 1])
    ax.set_xlabel(x_label)
    ax.set_ylabel('watch percentage')
    ax.set_xscale('log')


if __name__ == '__main__':
    with open('global_params/global_parameters_train.txt', 'r') as fin:
        duration_gap = read_as_float_array(fin.readline().rstrip(), delimiter=',')
        mean_watch_prec = read_as_float_array(fin.readline().rstrip(), delimiter=',')
        median_watch_prec = read_as_float_array(fin.readline().rstrip(), delimiter=',')
        upper_watch_prec = read_as_float_array(fin.readline().rstrip(), delimiter=',')
        lower_watch_prec = read_as_float_array(fin.readline().rstrip(), delimiter=',')
    bin_num = len(mean_watch_prec)

    input_loc = 'predict_results/predict_matrix_backup.txt'
    # input_loc = 'norm_predict_results/predict_incomplete_org.txt'
    ape_matrix = []
    r2_matrix = []

    dur_err = []
    dur_r2 = []
    dur_err_norm = []

    content_err = []
    content_r2 = []
    content_err_norm = []

    topic_err = []
    topic_r2 = []
    topic_err_norm = []

    channel_err = []
    channel_r2 = []
    channel_err_norm = []

    c_t_err = []
    c_t_r2 = []
    c_t_err_norm = []

    u_c_err = []
    u_c_r2 = []
    u_c_err_norm = []

    u_t_err = []
    u_t_r2 = []
    u_t_err_norm = []

    u_c_t_err = []
    u_c_t_r2 = []
    u_c_t_err_norm = []

    true_dur_wp_tuple = []
    dur_dur_wp_tuple = []
    content_dur_wp_tuple = []
    topic_dur_wp_tuple = []
    channel_dur_wp_tuple = []
    c_t_dur_wp_tuple = []
    u_c_dur_wp_tuple = []
    u_t_dur_wp_tuple = []
    u_c_t_dur_wp_tuple = []

    with open(input_loc, 'r') as fin:
        fin.readline()
        for line in fin:
            vid, duration, _, _, _, _, _, _, true_wp, d_wp, dc_wp, dt_wp, du_wp, dct_wp, duc_wp, dut_wp, duct_wp = line.rstrip().split('\t')
            # vid, duration, _, _, _, _, _, _, true_wp, wp_percentile, d_wp, dc_wp, du_wp, duc_wp, dut_wp, duct_wp = line.rstrip().split('\t')
            # vid, duration, _, _, _, _, _, _, true_wp, wp_percentile, d_wp, dc_wp, du_wp, duct_wp = line.rstrip().split('\t')
            duration = int(duration)
            true_wp = float(true_wp)

            # bin_idx = min(np.sum(duration_gap < duration), bin_num - 1)
            # mean_wp = get_mean_prec(bin_idx)
            # median_wp = get_median_prec(bin_idx)
            # upper_wp = get_upper_bound(bin_idx)
            # lower_wp = get_lower_bound(bin_idx)
            # bin_range = upper_wp - lower_wp

            true_dur_wp_tuple.append((duration, true_wp))

            if d_wp != 'NA':
                dur_wp = float(d_wp)
                dur_dur_wp_tuple.append((duration, dur_wp))
                dur_err.append(abs(true_wp - dur_wp) * 100)
                # dur_err_norm.append(abs(true_wp - dur_wp) / bin_range * 100)

            if dc_wp != 'NA':
                content_wp = float(dc_wp)
                content_dur_wp_tuple.append((duration, content_wp))
                content_err.append(abs(true_wp-content_wp) * 100)
                # content_err_norm.append(abs(true_wp-content_wp)/bin_range*100)

            if du_wp != 'NA':
                channel_wp = float(du_wp)
                channel_dur_wp_tuple.append((duration, channel_wp))
                channel_err.append(abs(true_wp - channel_wp) * 100)
                # channel_err_norm.append(abs(true_wp - channel_wp) / bin_range * 100)

            if dt_wp != 'NA':
                topic_wp = float(dt_wp)
                topic_dur_wp_tuple.append((duration, topic_wp))
                topic_err.append(abs(true_wp-topic_wp) * 100)
                # topic_err_norm.append(abs(true_wp-topic_wp)/bin_range*100)

            if dct_wp != 'NA':
                dct_wp = float(dct_wp)
                c_t_dur_wp_tuple.append((duration, dct_wp))
                c_t_err.append(abs(true_wp - dct_wp) * 100)
                # c_t_err_norm.append(abs(true_wp - c_t_wp) / bin_range * 100)

            if duc_wp != 'NA':
                duc_wp = float(duc_wp)
                u_c_dur_wp_tuple.append((duration, duc_wp))
                u_c_err.append(abs(true_wp - duc_wp) * 100)
                # u_c_err_norm.append(abs(true_wp - u_c_wp) / bin_range * 100)

            if dut_wp != 'NA':
                dut_wp = float(dut_wp)
                u_t_dur_wp_tuple.append((duration, dut_wp))
                u_t_err.append(abs(true_wp - dut_wp) * 100)
                # u_t_err_norm.append(abs(true_wp - u_t_wp) / bin_range * 100)

            if duct_wp != 'NA':
                duct_wp = float(duct_wp)
                u_c_t_dur_wp_tuple.append((duration, duct_wp))
                u_c_t_err.append(abs(true_wp - duct_wp) * 100)
                # u_c_t_err_norm.append(abs(true_wp - u_c_t_wp) / bin_range * 100)

    ape_matrix.append(dur_err)
    # dur_r2 = r2_score([x[1] for x in true_dur_wp_tuple], [x[1] for x in dur_dur_wp_tuple])
    # r2_matrix.append(dur_r2)

    ape_matrix.append(content_err)
    # content_r2 = r2_score([x[1] for x in true_dur_wp_tuple], [x[1] for x in content_dur_wp_tuple])
    # r2_matrix.append(content_r2)

    ape_matrix.append(topic_err)
    # topic_r2 = r2_score([x[1] for x in true_dur_wp_tuple], [x[1] for x in topic_dur_wp_tuple])
    # r2_matrix.append(topic_r2)
    #
    ape_matrix.append(c_t_err)
    # c_t_r2 = r2_score([x[1] for x in true_dur_wp_tuple], [x[1] for x in c_t_dur_wp_tuple])
    # r2_matrix.append(c_t_r2)
    #
    ape_matrix.append(channel_err)
    # channel_r2 = r2_score([x[1] for x in true_dur_wp_tuple], [x[1] for x in channel_dur_wp_tuple])
    # r2_matrix.append(channel_r2)
    #
    ape_matrix.append(u_c_err)
    # u_c_r2 = r2_score([x[1] for x in true_dur_wp_tuple], [x[1] for x in u_c_dur_wp_tuple])
    # r2_matrix.append(u_c_r2)
    #
    ape_matrix.append(u_t_err)
    # u_t_r2 = r2_score([x[1] for x in true_dur_wp_tuple], [x[1] for x in u_t_dur_wp_tuple])
    # r2_matrix.append(u_t_r2)
    #
    ape_matrix.append(u_c_t_err)
    # u_c_t_r2 = r2_score([x[1] for x in true_dur_wp_tuple], [x[1] for x in u_c_t_dur_wp_tuple])
    # r2_matrix.append(u_c_t_r2)

    # print('r2 of duration', r2_score([x[1] for x in true_dur_wp_tuple], [x[1] for x in dur_dur_wp_tuple]))
    # print('r2 of content', r2_score([x[1] for x in true_dur_wp_tuple], [x[1] for x in content_dur_wp_tuple]))
    # print('r2 of topic', r2_score([x[1] for x in true_dur_wp_tuple], [x[1] for x in topic_dur_wp_tuple]))
    # print('r2 of content+topic', r2_score([x[1] for x in true_dur_wp_tuple], [x[1] for x in c_t_dur_wp_tuple]))
    # print('r2 of channel', r2_score([x[1] for x in true_dur_wp_tuple], [x[1] for x in channel_dur_wp_tuple]))
    # print('r2 of channel+content', r2_score([x[1] for x in true_dur_wp_tuple], [x[1] for x in u_c_dur_wp_tuple]))
    # print('r2 of channel+topic', r2_score([x[1] for x in true_dur_wp_tuple], [x[1] for x in u_t_dur_wp_tuple]))
    # print('r2 of channel+content+topic', r2_score([x[1] for x in true_dur_wp_tuple], [x[1] for x in u_c_t_dur_wp_tuple]))

    print('Number of test videos: {0}'.format(len(dur_err)))

    fig = plt.figure(figsize=(8, 6))
    plot_performance_comp(ape_matrix, 0)
    # plot_r2_comp(r2_matrix, 1)

    # fig = plt.figure(figsize=(14, 10))
    # plot_dur_wp_dist(true_dur_wp_tuple, 1, 'true dist')
    # plot_dur_wp_dist(dur_dur_wp_tuple, 2, 'duration dist')
    # plot_dur_wp_dist(content_dur_wp_tuple, 3, 'content dist')
    # # plot_dur_wp_dist(topic_dur_wp_tuple, 4, 'topic dist')
    # # plot_dur_wp_dist(c_t_dur_wp_tuple, 5, 'content+topic dist')
    # plot_dur_wp_dist(channel_dur_wp_tuple, 6, 'channel dist')
    # # plot_dur_wp_dist(u_c_dur_wp_tuple, 7, 'channel+content dist')
    # # plot_dur_wp_dist(u_t_dur_wp_tuple, 8, 'channel+topic dist')
    # # plot_dur_wp_dist(u_c_t_dur_wp_tuple, 9, 'channel+content+topic dist')

    plt.tight_layout()
    plt.show()
