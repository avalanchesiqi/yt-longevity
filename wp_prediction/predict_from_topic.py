from __future__ import print_function, division
import sys
import os
import numpy as np
from collections import defaultdict
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


def weighted_avg(tuple_arr):
    data = np.array([x[0] for x in tuple_arr])
    weight = np.array([x[1] for x in tuple_arr])
    return np.sum(data*weight)/np.sum(weight)


if __name__ == '__main__':
    fig = plt.figure()

    with open('global_params/global_parameters_train.txt', 'r') as fin:
        duration_gap = read_as_float_array(fin.readline().rstrip(), delimiter=',')
        mean_watch_prec = read_as_float_array(fin.readline().rstrip(), delimiter=',')
        median_watch_prec = read_as_float_array(fin.readline().rstrip(), delimiter=',')
        upper_watch_prec = read_as_float_array(fin.readline().rstrip(), delimiter=',')
        lower_watch_prec = read_as_float_array(fin.readline().rstrip(), delimiter=',')
    bin_num = len(mean_watch_prec)

    train_doc = '../../data/production_data/wp_prediction/train_data'
    train_params_loc = 'topic_coef_dir'
    test_doc = '../../data/production_data/wp_prediction/test_data'
    output_path = open('predict_results/predict_dur_topic.txt', 'w')
    # print('>>> Load world topic count! Number of topics: {0}'.format(len(world_topic_count)))

    dur_topic_errors = []
    bagging_models = []
    bagging_models_stats = []

    c1 = 0
    num_model = 1024
    for i in xrange(num_model):
        mini_model = {}
        model_stats = {}
        with open(os.path.join(train_params_loc, 'topic_coef_{0}.txt'.format(i)), 'r') as fin:
            for line in fin:
                topic, coef, occurrence = line.rstrip().split(',')
                mini_model[topic] = float(coef)
                model_stats[topic] = int(occurrence)
        # print('topic in mini model {0}: {1}'.format(i, len(mini_model)))
        bagging_models.append(mini_model)
        bagging_models_stats.append(model_stats)
        c1 += 1
        if c1 > num_model:
            break

    for subdir, _, files in os.walk(test_doc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                for line in fin:
                    vid, duration, definition, category_id, channel_id, topics, total_view, true_wp = line.rstrip().split('\t')
                    duration = int(duration)
                    true_wp = float(true_wp)
                    bin_idx = min(np.sum(duration_gap < duration), bin_num-1)
                    mean_wp = get_mean_prec(bin_idx)
                    median_wp = get_median_prec(bin_idx)
                    upper_wp = get_upper_bound(bin_idx)
                    lower_wp = get_lower_bound(bin_idx)
                    bin_range = upper_wp - lower_wp

                    topics = topics.split(',')
                    bagging_results_vector = []
                    for model_idx, mini_model in enumerate(bagging_models):
                        pred_wp = 0
                        row = np.zeros(len(mini_model))
                        sampled = True
                        for topic in topics:
                            if topic in mini_model:
                                pred_wp += mini_model[topic]
                            else:
                                sampled = False
                        if sampled:
                            model_stats = bagging_models_stats[model_idx]
                            weight = np.inf
                            for topic in topics:
                                if model_stats[topic] < weight:
                                    weight = model_stats[topic]
                            pred_wp = pred_wp*bin_range + median_wp
                            if pred_wp > upper_wp:
                                pred_wp = upper_wp
                            elif pred_wp < lower_wp:
                                pred_wp = lower_wp
                            bagging_results_vector.append((pred_wp, weight))

                    if len(bagging_results_vector) == 0:
                        final_topic_wp = mean_wp
                    else:
                        final_topic_wp = np.mean([x[0] for x in bagging_results_vector])

                        # print('{6}: {8}-{7}; true wp: {0:.4f}; dur wp: {1:.4f}; topic wp: {2:.4f} / {3} - {4} - {5}'
                        #       .format(true_wp, mean_wp, final_topic_wp, len(topics), len(bagging_results_vector),
                        #               sorted(bagging_results_vector, key=lambda tup: tup[1], reverse=True), vid, total_view, duration))
                        # output_path.write('{0}\t{1}\t{2}\t{3}\n'.format(vid, watch_prec, dur_predict, final_watch_prec))

                    dur_wp = mean_wp
                    topic_wp = final_topic_wp
                    cat_wp = 'NA'
                    user_wp = 'NA'
                    output_path.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(line.rstrip(), dur_wp, topic_wp, cat_wp, user_wp))

    output_path.close()
