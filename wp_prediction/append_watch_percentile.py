from __future__ import print_function, division
import sys
import os
import numpy as np
import cPickle as pickle
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def get_percentile(duration, true_wp):
    bin_idx = np.sum(duration_split_points < duration)
    duration_bin = dur_engage_map[bin_idx]
    wp_percentile = np.sum(np.array(duration_bin) < true_wp) / 1000
    return wp_percentile


def get_wp(duration, percentile):
    bin_idx = np.sum(duration_split_points < duration)
    duration_bin = dur_engage_map[bin_idx]
    percentile = int(round(percentile*1000))
    wp_percentile = duration_bin[percentile]
    return wp_percentile


def plot_model(ax, pred_train_wp, pred_test_wp):
    train_mae = mean_absolute_error(train_wp, pred_train_wp)
    train_r2 = r2_score(train_wp, pred_train_wp)
    test_mae = mean_absolute_error(test_wp, pred_test_wp)
    test_r2 = r2_score(test_wp, pred_test_wp)

    show_train = False
    if show_train:
        ax.scatter(train_dur, train_wp, c='k', marker='.', label='Train MAE: {0:.4f}'.format(train_mae))
        ax.scatter(train_dur, pred_train_wp, c='g', marker='x', label='Predicted train')
        for i in xrange(len(train_dur)):
            ax.plot((train_dur[i], train_dur[i]), (train_wp[i], pred_train_wp[i]), 'k--')

    show_test = True
    if show_test:
        ax.scatter(test_dur, test_wp, c='k', marker='.', label='Test MAE: {0:.4f}'.format(test_mae))
        ax.scatter(test_dur, pred_test_wp, c='r', marker='+', label='Predicted test')
        for i in xrange(len(test_dur)):
            ax.plot((test_dur[i], test_dur[i]), (test_wp[i], pred_test_wp[i]), 'k--')

    ax.set_ylim([0, 1])
    ax.set_xlabel('Video duration (sec)', fontsize=20)
    ax.set_ylabel('Watch percentage', fontsize=20)
    ax.set_xscale('log')
    ax.legend()

    print('>>> Train MAE: {0}'.format(train_mae))
    print('>>> Train R2: {0}'.format(train_r2))
    print('>>> Test MAE: {0}'.format(test_mae))
    print('>>> Test R2: {0}'.format(test_r2))
    print('='*79)
    print()


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    dur_engage_str_map = pickle.load(open('dur_engage_map.p', 'rb'))
    dur_engage_map = {key: list(map(float, value.split(','))) for key, value in dur_engage_str_map.items()}

    duration_split_points = np.array(dur_engage_map['duration'])

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    input_loc = '../../data/production_data/random_langs/train_data'
    output_loc = '../../data/production_data/random_norm/train_data'

    for subdir, _, files in os.walk(input_loc):
        for f in files:
            fout = open(os.path.join(output_loc, f), 'w')
            fout.write('vid\tduration\tdefinition\tcategory\tlang\tchannel\ttopics\ttotal_view\ttrue_wp\twp_percentile\n')
            with open(os.path.join(subdir, f), 'r') as fin:
                fin.readline()
                for line in fin:
                    _, duration, _, _, _, _, _, _, true_wp = line.rstrip().split('\t')
                    duration = int(duration)
                    true_wp = float(true_wp)

                    # get correct bin idx
                    wp_percentile = get_percentile(duration, true_wp)
                    fout.write('{0}\t{1}\n'.format(line.rstrip(), wp_percentile))
            fout.close()

    to_plot = False
    if to_plot:
        train_dur = [660, 231, 76, 220, 175, 356, 645, 401, 161, 1142, 78, 1002, 1648, 161, 81, 79, 346, 345, 98, 492, 192, 73, 495, 485, 365, 77, 256, 304, 263, 84]
        train_wp = [0.172932512375, 0.379171176046, 0.591259398495, 0.26580422362, 0.522098081023, 0.125698491858, 0.220291141488, 0.246381533603, 0.490464011691, 0.0499194395797, 0.302427845906, 0.153433443191, 0.0875989425515, 0.348230860492, 0.542195767196, 0.577036576843, 0.327231856133, 0.226347341012, 0.553324555629, 0.419506927251, 0.489682539682, 0.660492305526, 0.219430485762, 0.286963515391, 0.214892477233, 0.586443296481, 0.194374413696, 0.387397627364, 0.492627283687, 0.491836734694]

        test_dur = [295, 28, 21, 598, 453]
        test_wp = [0.2017296905, 0.943663292088, 0.888779608227, 0.294544173281, 0.190744042565]

        # encoded model
        encoded_wp = [get_percentile(x, y) for x, y in zip(train_dur, train_wp)]
        # train on lin duration
        ridge_encode_lin = Ridge(fit_intercept=True)
        ridge_encode_lin.fit(np.array(train_dur).reshape(-1, 1), np.array(encoded_wp).reshape(-1, 1))
        encode_pred_train_wp_lin = [get_wp(x, ridge_encode_lin.predict(x)) for x in train_dur]
        encode_pred_test_wp_lin = [get_wp(x, ridge_encode_lin.predict(x)) for x in test_dur]

        # train on log duration
        ridge_encode_log = Ridge(fit_intercept=True)
        ridge_encode_log.fit(np.log10(np.array(train_dur)).reshape(-1, 1), np.array(encoded_wp).reshape(-1, 1))
        encode_pred_train_wp_log = [get_wp(x, ridge_encode_log.predict(np.log10(x))) for x in train_dur]
        encode_pred_test_wp_log = [get_wp(x, ridge_encode_log.predict(np.log10(x))) for x in test_dur]

        # raw model
        # train on lin duration
        ridge_model_lin = Ridge(fit_intercept=True)
        ridge_model_lin.fit(np.array(train_dur).reshape(-1, 1), np.array(train_wp).reshape(-1, 1))
        raw_pred_train_wp_lin = [ridge_model_lin.predict(x)[0][0] for x in train_dur]
        raw_pred_test_wp_lin = [ridge_model_lin.predict(x)[0][0] for x in test_dur]

        # train on log duration
        ridge_model_log = Ridge(fit_intercept=True)
        ridge_model_log.fit(np.log10(np.array(train_dur)).reshape(-1, 1), np.array(train_wp).reshape(-1, 1))
        raw_pred_train_wp_log = [ridge_model_log.predict(np.log10(x))[0][0] for x in train_dur]
        raw_pred_test_wp_log = [ridge_model_log.predict(np.log10(x))[0][0] for x in test_dur]

        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(221)
        print('>>> Linear encoded:')
        plot_model(ax1, encode_pred_train_wp_lin, encode_pred_test_wp_lin)

        ax2 = fig.add_subplot(222)
        print('>>> Log encoded:')
        plot_model(ax2, encode_pred_train_wp_log, encode_pred_test_wp_log)

        ax3 = fig.add_subplot(223)
        print('>>> Linear raw:')
        plot_model(ax3, raw_pred_train_wp_lin, raw_pred_test_wp_lin)

        ax4 = fig.add_subplot(224)
        print('>>> Log raw:')
        plot_model(ax4, raw_pred_train_wp_log, raw_pred_test_wp_log)

        plt.tight_layout()
        plt.show()
