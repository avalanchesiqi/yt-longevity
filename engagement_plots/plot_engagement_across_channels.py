#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Scripts to plot relative engagement and watch percentage of two channels.

quality video: UCpw2gh99XM6Mwsbksv0feEg -- Moojiji
junk video: UC15CpXEtQ4DH6UCE7kNcQOA -- Jon Drinks Water 
"""

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # == == == == == == == == Part 1: Load dataset == == == == == == == ==
    quality_channel = {'channel_id': 'UCpw2gh99XM6Mwsbksv0feEg',
                       'channel_title': 'Moojiji',
                       'durations': [10867, 5228, 359, 11781, 3156, 11180, 12336, 9061, 982, 1239, 12283, 9867, 3726, 5210, 10302, 9050, 336, 433, 172, 211, 8375, 10390, 748, 3925, 3277, 5964, 9928],
                       'watch_percentage': [0.150914915744, 0.232985035277, 0.626052812322, 0.141905859914, 0.382841207634, 0.172787866772, 0.134773059716, 0.175560736881, 0.604992804237, 0.433635423391, 0.13834705196, 0.199265986449, 0.232638353976, 0.248497956096, 0.191515121744, 0.168735143458, 0.792272135988, 0.555906367157, 0.856781934141, 0.782353663793, 0.141189906727, 0.158712041234, 0.507308940463, 0.201337686282, 0.37461913674, 0.266802268849, 0.206685378021],
                       'relative_engagement': [0.768, 0.829, 0.906, 0.879, 0.946, 0.901, 0.913, 0.886, 0.982, 0.84, 0.895, 0.951, 0.738, 0.868, 0.944, 0.877, 0.995, 0.834, 0.993, 0.98, 0.801, 0.903, 0.849, 0.645, 0.945, 0.925, 0.956]}
    junk_channel = {'channel_id': 'UC15CpXEtQ4DH6UCE7kNcQOA',
                    'channel_title': 'Jon Drinks Water',
                    'durations': [423, 14, 727, 21, 502, 13, 22, 17, 31, 568, 413, 302, 18, 23, 14, 547, 10, 20, 13, 19, 17, 19, 21, 20, 300, 891, 10, 11, 11, 10],
                    'watch_percentage': [0.249396370293, 0.869557823129, 0.255053551064, 0.856096284668, 0.274539621456, 0.889561881688, 0.815594974132, 0.804653204565, 0.777642523838, 0.245447623604, 0.361357094752, 0.348108229335, 0.803921568628, 0.831873036866, 0.856934610578, 0.248353010431, 0.871767810026, 0.832000000001, 0.885473262796, 0.870832050888, 0.894548063127, 0.86365914787, 0.758818342152, 0.860179640718, 0.26406984127, 0.189818916914, 0.929589632829, 0.887833559476, 0.882003710576, 0.944951923076],
                    'relative_engagement': [0.167, 0.241, 0.279, 0.403, 0.239, 0.248, 0.28, 0.161, 0.31, 0.202, 0.39, 0.3, 0.174, 0.352, 0.21, 0.209, 0.146, 0.276, 0.238, 0.38, 0.388, 0.357, 0.175, 0.354, 0.138, 0.192, 0.254, 0.203, 0.192, 0.286]}

    print('Quality channel: {0}, mean watch percentage: {1:.4f}, mean relative engagement: {2: .4f}'
          .format(quality_channel['channel_title'], np.mean(quality_channel['watch_percentage']), np.mean(quality_channel['relative_engagement'])))

    print('Junk channel: {0}, mean watch percentage: {1:.4f}, mean relative engagement: {2: .4f}'
          .format(junk_channel['channel_title'], np.mean(junk_channel['watch_percentage']), np.mean(junk_channel['relative_engagement'])))

    # Plot examples
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    ax.plot((0, 1), (0, 1), 'k--', zorder=1)

    plot_ax1 = True
    if plot_ax1:
        ax.scatter(quality_channel['watch_percentage'], quality_channel['relative_engagement'], marker='o',
                   facecolors='none', edgecolors='r', s=60*np.log10(quality_channel['durations']),
                   label=quality_channel['channel_title'])

    plot_ax2 = True
    if plot_ax2:
        ax.scatter(junk_channel['watch_percentage'], junk_channel['relative_engagement'], marker='^',
                   facecolors='none', edgecolors='b', s=60*np.log10(junk_channel['durations']),
                   label=junk_channel['channel_title'])

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('watch percentage '+r'$\bar \mu_{30}$', fontsize=24)
    ax.set_ylabel('relative engagement '+r'$\bar \eta_{30}$', fontsize=24)
    ax.legend(loc='lower left', scatterpoints=2, frameon=True, fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)

    plt.tight_layout()
    plt.show()
