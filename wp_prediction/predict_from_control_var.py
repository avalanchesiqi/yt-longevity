import numpy as np
import os


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


def get_mean_prec(dur):
    return mean_watch_prec[np.sum(duration_gap < dur)]


if __name__ == '__main__':
    with open('global_params/global_parameters_train.txt', 'r') as fin:
        duration_gap = read_as_float_array(fin.readline().rstrip(), delimiter=',')
        mean_watch_prec = read_as_float_array(fin.readline().rstrip(), delimiter=',')

    output_path = open('predict_results/predict_dur_topic.txt', 'w')
    test_loc = '../../data/production_data/wp_prediction/test_data'
    for subdir, _, files in os.walk(test_loc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                for line in fin:
                    vid, duration, definition, category_id, channel_id, topics, total_view, true_wp = line.rstrip().split('\t')
                    dur_wp = get_mean_prec(int(duration))
                    topic_wp = 'NA'
                    cat_wp = 'NA'
                    user_wp = 'NA'
                    output_path.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(line.rstrip(), dur_wp, topic_wp, cat_wp, user_wp))
    output_path.close()

    print('input duration: {0}, predict watch prec: {1}'.format(10273, get_mean_prec(10273)))
    print('input duration: {0}, predict watch prec: {1}'.format(1204, get_mean_prec(1204)))
    print('input duration: {0}, predict watch prec: {1}'.format(20584, get_mean_prec(20584)))
    print('input duration: {0}, predict watch prec: {1}'.format(16333, get_mean_prec(16333)))
    print('input duration: {0}, predict watch prec: {1}'.format(2000, get_mean_prec(2000)))
