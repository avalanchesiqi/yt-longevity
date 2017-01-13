#!/usr/bin/python

# Usage example:
# python plot_sampling_kendall_tau.py --sample='<sample_file>' --complete='<complete_file>'

import os
import argparse
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = '../'


if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sample', help='input file path of sample data, relative to base dir', required=True)
    parser.add_argument('-c', '--complete', help='output file path of complete data, relative to base dir', required=True)
    args = parser.parse_args()

    sample_path = os.path.join(BASE_DIR, args.sample)
    complete_path = os.path.join(BASE_DIR, args.complete)

    sample_ranking = []
    with open(sample_path, 'r') as f:
        for line in f:
            sample_ranking.append(int(line.rstrip().split()[1]))

    complete_ranking = []
    with open(complete_path, 'r') as f:
        for line in f:
            complete_ranking.append(int(line.rstrip().split()[1]))

    taus = []
    p_values = []
    for i in xrange(10, 1001, 10):
        tau, p_value = stats.kendalltau(sample_ranking[:i], complete_ranking[:i])
        taus.append(tau)
        p_values.append(p_value)

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    fig, ax1 = plt.subplots(1, 1)
    ax1.plot(np.arange(10, 1001, 10), taus)

    ax1.set_ylim(ymin=0)
    ax1.set_xlabel('n')
    ax1.set_ylabel('tau')
    ax1.set_title('Figure 1: Most tweeted videos - Firehose and Streaming API')

    plt.show()
