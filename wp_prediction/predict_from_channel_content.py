from __future__ import division, print_function
import os
import pandas as pd
import numpy as np
from operator import itemgetter
import cPickle as pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVR
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
            channel_content_vid_wp_dict[vid] = 'NA'


def remove_invalid_prediction(arr):
    arr[arr > 1] = 1
    arr[arr < 0] = 0
    return arr


if __name__ == '__main__':
    with open('global_params/global_parameters_train.txt', 'r') as fin:
        duration_gap = read_as_float_array(fin.readline().rstrip(), delimiter=',')
        mean_watch_prec = read_as_float_array(fin.readline().rstrip(), delimiter=',')

    output_path = 'predict_results/vid_channel_content.p'
    channel_content_vid_wp_dict = {}

    train_loc = '../../data/production_data/random_channel/train_data'
    test_loc = '../../data/production_data/random_channel/test_data'

    for subdir, _, files in os.walk(test_loc):
        for f in files:
            # if we have observed this channel before
            if os.path.exists(os.path.join(train_loc, f)):
                # get past success
                with open(os.path.join(test_loc, f), 'r') as fin1:
                    train_lines = fin1.read().splitlines()
                    if len(train_lines) > 5:
                        # predict from past success and content
                        train_df = pd.read_csv(os.path.join(train_loc, f), sep='\t', header=None,
                                               names=['vid', 'duration', 'definition', 'category', 'lang', 'channel',
                                                      'topics', 'total_view', 'watch_percentage'],
                                               dtype={'duration': int, 'watch_percentage': np.float32})
                        test_df = pd.read_csv(os.path.join(test_loc, f), sep='\t', header=None,
                                              names=['vid', 'duration', 'definition', 'category', 'lang', 'channel',
                                                     'topics', 'total_view', 'watch_percentage'],
                                              dtype={'duration': int, 'watch_percentage': np.float32})
                        train_num = train_df.shape[0]
                        data_df = train_df.append(test_df)

                        train_df = data_df[:train_num]
                        test_df = data_df[train_num:]

                        categorical_values = np.array(data_df[['definition', 'category', 'lang']])

                        # do the first column
                        enc_label = LabelEncoder()
                        categorical_data = enc_label.fit_transform(categorical_values[:, 0])

                        # do the others
                        for i in range(1, categorical_values.shape[1]):
                            enc_label = LabelEncoder()
                            categorical_data = np.column_stack((categorical_data, enc_label.fit_transform(categorical_values[:, i])))

                        categorical_values = categorical_data.astype(float)

                        # if you have only integers then you can skip the above part from do the first column and uncomment the following line
                        # train_categorical_values = train_categorical_values.astype(float)

                        enc_onehot = OneHotEncoder()
                        cat_data = enc_onehot.fit_transform(categorical_values)

                        cat_df = pd.DataFrame(cat_data.toarray())

                        data_ndarray = np.column_stack((cat_df, np.log10(data_df['duration']), data_df['watch_percentage']))

                        train_x = data_ndarray[:train_num, :-1]
                        train_y = data_ndarray[:train_num, -1]
                        test_x = data_ndarray[train_num:, :-1]
                        test_y = data_ndarray[train_num:, -1]

                        svm_model = SVR(C=1.0, epsilon=0.2).fit(train_x, train_y)

                        pred_test_y = svm_model.predict(test_x)
                        pred_test_y = remove_invalid_prediction(pred_test_y)

                        test_df['u_c_wp'] = pred_test_y
                        pred_result = test_df[['vid', 'u_c_wp']]

                        channel_content_vid_wp_dict.update(test_df.set_index('vid')['u_c_wp'].to_dict())

                        # crunch_data = [itemgetter(*x.split('\t'))([1, 2, 3, 4, 8]) for x in train_lines]
                        # m = len(crunch_data)
                        # train_x = np.array([np.log10(int(x[0])) for x in crunch_data]).reshape(-1, 1)
                        # train_y = np.array([float(x[1]) for x in crunch_data]).reshape(-1, 1)
                        #
                        # ridge_model = Ridge(fit_intercept=True)
                        # ridge_model.fit(train_x, train_y)
                        #
                        # with open(os.path.join(subdir, f), 'r') as fin2:
                        #     for line in fin2:
                        #         vid, duration, _, _, _, _, _, _, true_wp = line.rstrip().split('\t')
                        #         duration = np.log10(int(duration))
                        #         channel_wp = ridge_model.predict(duration)[0][0]
                        #         if channel_wp > 1:
                        #             channel_wp = 1
                        #         elif channel_wp < 0:
                        #             channel_wp = 0
                        #         channel_vid_wp_dict[vid] = channel_wp
                    else:
                        predict_from_duration(os.path.join(subdir, f))
            # if not, predict from duration
            else:
                predict_from_duration(os.path.join(subdir, f))

    # write to txt file
    print('>>> Number of videos in final test result dict: {0}'.format(len(channel_content_vid_wp_dict)))
    pickle.dump(channel_content_vid_wp_dict, open(output_path, 'wb'))
