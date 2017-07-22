from __future__ import division, print_function
import os
import numpy as np
import cPickle as pickle
from sklearn.linear_model import Ridge


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


def predict_from_duration(filepath):
    with open(filepath, 'r') as fin:
        for line in fin:
            vid, duration, _, _, _, _, _, _, true_wp = line.rstrip().split('\t')
            duration = int(duration)
            dur_wp = get_mean_prec(duration)
            channel_vid_wp_dict[vid] = dur_wp


if __name__ == '__main__':
    with open('global_params/global_parameters_train.txt', 'r') as fin:
        duration_gap = read_as_float_array(fin.readline().rstrip(), delimiter=',')
        mean_watch_prec = read_as_float_array(fin.readline().rstrip(), delimiter=',')

    output_path = 'predict_results/vid_channel.p'
    channel_vid_wp_dict = {}

    for subdir, _, files in os.walk('../../data/production_data/random_channel/test_data'):
        for f in files:
            # if we have observed this channel before
            if os.path.exists(os.path.join('../../data/production_data/random_channel/train_data', f)):
                # get past success
                with open(os.path.join('../../data/production_data/random_channel/train_data', f), 'r') as fin1:
                    train_lines = fin1.read().splitlines()
                    if len(train_lines) > 5:
                        # predict from past success
                        crunch_data = [(x.split('\t')[1], x.split('\t')[8]) for x in train_lines]
                        train_x = np.array([np.log10(int(x[0])) for x in crunch_data]).reshape(-1, 1)
                        train_y = np.array([float(x[1]) for x in crunch_data]).reshape(-1, 1)

                        ridge_model = Ridge(fit_intercept=True)
                        ridge_model.fit(train_x, train_y)

                        with open(os.path.join(subdir, f), 'r') as fin2:
                            for line in fin2:
                                vid, duration, _, _, _, _, _, _, true_wp = line.rstrip().split('\t')
                                duration = np.log10(int(duration))
                                channel_wp = ridge_model.predict(duration)[0][0]
                                if channel_wp > 1:
                                    channel_wp = 1
                                elif channel_wp < 0:
                                    channel_wp = 0
                                channel_vid_wp_dict[vid] = channel_wp
                    else:
                        predict_from_duration(os.path.join(subdir, f))
            # if not, predict from duration
            else:
                predict_from_duration(os.path.join(subdir, f))

    # write to txt file
    print('>>> Number of videos in final test result dict: {0}'.format(len(channel_vid_wp_dict)))
    pickle.dump(channel_vid_wp_dict, open(output_path, 'wb'))
