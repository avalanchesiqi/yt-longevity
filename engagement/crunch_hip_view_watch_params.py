#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os
from scipy import stats

# Extract global duration ~ watch percentage mapping from tweeted videos dataset

#  input: vid, duration, mu, theta, C, c, gamma, eta, exo, endo, viral, wp@30, wp@60, wp@90, wp@120
# output: vid, duration, mu, theta, C, c, gamma, eta, exo, exo_rank, endo, endo_rank, viral, viral_rank, wp@30, wp@60, wp@90, wp@120


if __name__ == '__main__':
    hip_view_path = './data/active_view_nonreg_params.txt'
    hip_view_rank_norm_path = './data/active_view_rank_norm_params.txt'
    hip_watch_path = './data/active_watch_nonreg_params.txt'
    hip_watch_rank_norm_path = './data/active_watch_rank_norm_params.txt'

    # rank normalize exo, endo, viral
    for in_path, out_path in zip([hip_view_path, hip_watch_path], [hip_view_rank_norm_path, hip_watch_rank_norm_path]):
        exo_list = []
        endo_list = []
        viral_list = []
        with open(in_path, 'r') as in_file:
            in_file.readline()
            for line in in_file:
                try:
                    vid, duration, mu, theta, C, c, gamma, eta, exo, endo, viral, wp30, wp60, wp90, wp120 = line.rstrip().split('\t')
                    exo = float(exo)
                    endo = float(endo)
                    viral = float(viral)
                    exo_list.append(exo)
                    endo_list.append(endo)
                    viral_list.append(viral)
                except:
                    break

        with open(out_path, 'w') as out_file:
            out_file.write('vid\tduration\tmu\ttheta\tC\tc\tgamma\teta\texo\texo_rank\tendo\tendo_rank\tviral\tviral_rank\twp@30\twp@60\twp@90\twp@120\n')
            with open(in_path, 'r') as in_file:
                in_file.readline()
                for line in in_file:
                    try:
                        vid, duration, mu, theta, C, c, gamma, eta, exo, endo, viral, wp30, wp60, wp90, wp120 = line.rstrip().split('\t')
                        exo = float(exo)
                        endo = float(endo)
                        viral = float(viral)
                        exo_rank = stats.percentileofscore(exo_list, exo)
                        endo_rank = stats.percentileofscore(endo_list, endo)
                        viral_rank = stats.percentileofscore(viral_list, viral)
                        to_write = [vid, duration, mu, theta, C, c, gamma, eta, exo, exo_rank, endo, endo_rank, viral, viral_rank, wp30, wp60, wp90, wp120]
                        out_file.write('{0}\n'.format('\t'.join(map(str, to_write))))
                    except:
                        break