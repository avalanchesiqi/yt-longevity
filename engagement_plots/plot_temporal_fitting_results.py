#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt

# Plot fitted engagement score parameters temporal


def get_float(text):
    return float(text.split(': ')[1])

if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    power_mae = []
    linear_mae = []
    constant_mae = []

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    input_loc = 'temporal_fitting.txt'
    vid_cnt = 0

    line_cnt = 0
    with open(input_loc, 'r') as fin:
        while True:
            try:
                vid_line = fin.readline().rstrip()
                line_cnt += 1
                if vid_line == '':
                    break
                _, _, power, linear, const, _ = vid_line.rstrip().split('\t', 5)
                power = float(power)
                linear = float(linear)
                const = float(const)

                power_mae.append(power)
                linear_mae.append(linear)
                constant_mae.append(const)
            except Exception as e:
                print(str(e))
                break

    # == == == == == == == == Part 3: Plot dataset == == == == == == == == #
    power_lo, power_hi = sms.DescrStatsW(power_mae).tconfint_mean()
    linear_lo, linear_hi = sms.DescrStatsW(linear_mae).tconfint_mean()
    constant_lo, constant_hi = sms.DescrStatsW(constant_mae).tconfint_mean()
    print('fitted video: {0}'.format(line_cnt))
    print('power law:', np.mean(power_mae), '+', np.std(power_mae))
    print('linear:', np.mean(linear_mae), '+', np.std(linear_mae))
    print('constant:', np.mean(constant_mae), '+', np.std(constant_mae))
    print('>>> Power-law fitting MAE mean: {0:.4f}, 95% confidence intervals: {1:.4f}'.format(np.mean(power_mae), (power_hi-power_lo)/2))
    print('>>> Linear fitting MAE mean: {0:.4f}, 95% confidence intervals: {1:.4f}'.format(np.mean(linear_mae), (linear_hi-linear_lo)/2))
    print('>>> Constant fitting MAE mean: {0:.4f}, 95% confidence intervals: {1:.4f}'.format(np.mean(constant_mae), (constant_hi-constant_lo)/2))

    fig, ax1 = plt.subplots(1, 1)
    ax1.boxplot([power_mae, linear_mae, constant_mae], labels=['power-law', 'linear', 'constant'],
                showfliers=False, showmeans=True)
    ax1.set_ylabel('MAE')

    means = [np.mean(power_mae), np.mean(linear_mae), np.mean(constant_mae)]
    mean_labels = [str(np.round(s, 4)) for s in means]
    pos = range(len(means))
    for tick, label in zip(pos, ax1.get_xticklabels()):
        ax1.text(pos[tick]+1, means[tick]+0.01, mean_labels[tick], horizontalalignment='center', color='k')

    plt.tight_layout()
    plt.show()
