from __future__ import print_function, division
import sys
import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn import metrics, linear_model
np.set_printoptions(suppress=True)


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


def training_process():
    # train the model
    training_matrix = None
    for video_vector in data_matrix:
        duration, topic_vector, true_wp = video_vector
        bin_idx = min(np.sum(duration_gap < duration), bin_num - 1)
        row = np.zeros(topic_cnt + 4)
        if len(topic_vector) != 0:
            for topic_idx in topic_vector:
                row[topic_idx] = 1
        row[-4] = duration
        row[-3] = get_upper_bound(bin_idx) - get_lower_bound(bin_idx)
        row[-2] = get_median_prec(bin_idx)
        row[-1] = (true_wp - row[-2]) / row[-3]
        if training_matrix is None:
            training_matrix = row
        else:
            training_matrix = np.vstack([training_matrix, row])

    print('>>> Numpy matrix shape: {0}'.format(training_matrix.shape))

    ridge_model = linear_model.Ridge(fit_intercept=False, solver='sparse_cg', alpha=0.5)
    ridge_model.fit(training_matrix[:, :-4], training_matrix[:, -1])
    print('>>> Finish learning in epoch{0}!'.format(epoch_cnt))
    print('>>> Topic in epoch{0}: {1}'.format(epoch_cnt, topic_cnt))
    topic_coef = ridge_model.coef_

    # mean absolute error in the same training set
    # duration_col = training_matrix[:, -4].reshape(-1, 1)
    true_wp_col = (training_matrix[:, -1] * training_matrix[:, -3] + training_matrix[:, -2]).reshape(-1, 1)
    pred_wp_col = (ridge_model.predict(training_matrix[:, :-4]) * training_matrix[:, -3] + training_matrix[:, -2]).reshape(-1, 1)
    # print(np.hstack([duration_col, true_wp_col, pred_wp_col]))
    print('>>> Train error in epoch{0}: {1}'.format(epoch_cnt, metrics.mean_absolute_error(true_wp_col, pred_wp_col)))

    fout = open(os.path.join(train_params_loc, 'topic_coef_{0}.txt'.format(epoch_cnt)), 'w')
    for topic, idx in topic_encoding_dict.items():
        fout.write('{0}, {1}, {2}\n'.format(topic, topic_coef[idx], topic_count_dict[topic_encoding_dict[topic]]))
    fout.close()


if __name__ == '__main__':
    category = 'activism'
    print('>>> Extract on data: {0}'.format(category))

    with open('global_params/global_parameters_{0}.txt'.format(category), 'r') as fin:
        duration_gap = read_as_float_array(fin.readline().rstrip(), delimiter=',')
        mean_watch_prec = read_as_float_array(fin.readline().rstrip(), delimiter=',')
        median_watch_prec = read_as_float_array(fin.readline().rstrip(), delimiter=',')
        upper_watch_prec = read_as_float_array(fin.readline().rstrip(), delimiter=',')
        lower_watch_prec = read_as_float_array(fin.readline().rstrip(), delimiter=',')
    bin_num = len(mean_watch_prec)

    train_doc = '../../data/production_data/wp_prediction/train_data/{0}'.format(category)
    train_params_loc = '../../data/production_data/wp_prediction/topic_coef_dir_{0}'.format(category)

    if not os.path.exists(train_params_loc):
        os.mkdir(train_params_loc)

    dur_topic_errors = []

    # train every 2,000 videos
    epoch_cnt = 0
    batch_cnt = 0
    topic_count_dict = defaultdict(int)
    topic_encoding_dict = {}
    topic_cnt = 0
    data_matrix = []

    for subdir, _, files in os.walk(train_doc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                for line in fin:
                    vid, duration, definition, category_id, channel_id, topics, total_view, true_wp = line.rstrip().split('\t')
                    duration = int(duration)
                    true_wp = float(true_wp)
                    topic_vector = []
                    topics = topics.split(',')
                    for topic in topics:
                        if topic not in topic_encoding_dict:
                            topic_encoding_dict[topic] = topic_cnt
                            topic_cnt += 1
                        topic_vector.append(topic_encoding_dict[topic])
                        topic_count_dict[topic_encoding_dict[topic]] += 1
                    data_matrix.append([duration, topic_vector, true_wp])
                    batch_cnt += 1

                    if batch_cnt == 2000:
                        training_process()

                        epoch_cnt += 1
                        batch_cnt = 0
                        topic_count_dict = defaultdict(int)
                        topic_encoding_dict = {}
                        topic_cnt = 0
                        data_matrix = []
                        print('='*79)
                        print()

    # leftover data
    if batch_cnt > 0:
        training_process()
