from __future__ import division, print_function
import pandas as pd
import numpy as np
import cPickle as pickle
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


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


def build_matrix(train_loc, test_loc):
    x = []
    y = []

    num_train = 0
    with open(train_loc, 'r') as fin:
        for line in fin:
            vid, duration, definition, category_id, lang, _, _, _, true_wp = line.rstrip().split('\t')
            x.append([np.log10(int(duration)), definition, category_id, lang])
            y.append(float(true_wp))
            num_train += 1

    with open(test_loc, 'r') as fin:
        for line in fin:
            vid, duration, definition, category_id, lang, _, _, _, true_wp = line.rstrip().split('\t')
            x.append([np.log10(int(duration)), definition, category_id, lang])
            y.append(float(true_wp))

    observations = np.array(x)
    labels = np.array(y)

    # # do the first column
    # cat_observations = observations[:, 1]
    # # enc_label = LabelEncoder()
    # # cat_observations = enc_label.fit_transform(observations[:, 0])
    #
    # # do the others
    # for i in range(1, observations.shape[1]):
    #     enc_label = LabelEncoder()
    #     cat_observations = np.column_stack((cat_observations, enc_label.fit_transform(observations[:, i])))
    #     print(enc_label.classes_)
    #
    # cat_observations_values = cat_observations.astype(float)
    #
    # enc_onehot = OneHotEncoder()
    # cat_observations_data = enc_onehot.fit_transform(cat_observations_values)
    # cat_observations_data = cat_observations_data.toarray()
    #
    # # # add control variable
    # cat_observations_data = np.column_stack((cat_observations_data, observations[:, 0]))

    train_x = observations[:num_train, :]
    test_x = labels[:num_train]
    train_y = observations[num_train:, :]
    test_y = labels[num_train:]

    return train_x, test_x, train_y, test_y


def remove_invalid_prediction(arr):
    arr[arr > 1] = 1
    arr[arr < 0] = 0
    return arr


if __name__ == '__main__':
    categories = ['activism', 'auto', 'comedy', 'education', 'entertainment', 'film', 'gaming', 'howto', 'movie', 'music', 'news', 'people', 'pets', 'science', 'show', 'sports', 'trailer', 'travel']
    pred_dict = {}

    for category in categories:
        print('>>> Train on category {0}'.format(category))
        train_loc = '../../data/production_data/random_langs/train_data/{0}.txt'.format(category)
        test_loc = '../../data/production_data/random_langs/test_data/{0}.txt'.format(category)

        train_df = pd.read_csv(train_loc, sep='\t', header=None,
                               names=['vid', 'duration', 'definition', 'category', 'lang', 'channel', 'topics', 'total_view', 'watch_percentage'],
                               dtype={'duration': int, 'definition': int, 'category': int, 'watch_percentage': np.float32})
        test_df = pd.read_csv(test_loc, sep='\t', header=None,
                              names=['vid', 'duration', 'definition', 'category', 'lang', 'channel', 'topics', 'total_view', 'watch_percentage'],
                              dtype={'duration': int, 'definition': int, 'category': int, 'watch_percentage': np.float32})

        train_num = train_df.shape[0]
        data_df = train_df.append(test_df)

        data_df['duration'] = np.log10(data_df['duration'])

        enc_label = LabelEncoder()
        data_df['lang'] = enc_label.fit_transform(data_df['lang'])

        train_df = data_df[:train_num]
        test_df = data_df[train_num:]

        cols = ['duration', 'definition', 'category', 'lang']

        rf_regressor = RandomForestRegressor(n_estimators=10, min_samples_leaf=20, random_state=42)
        rf_regressor.fit(train_df[cols], train_df.watch_percentage)

        print('>>> Feature importances of duration, definition, category, language: {0}'.format(rf_regressor.feature_importances_))
        print('>>> Number of features: {0}'.format(rf_regressor.n_features_))

        pred_train_y = rf_regressor.predict(train_df[cols])
        pred_train_y = remove_invalid_prediction(pred_train_y)

        pred_test_y = rf_regressor.predict(test_df[cols])
        pred_test_y = remove_invalid_prediction(pred_test_y)

        test_df['content_wp'] = pred_test_y
        pred_result = test_df[['vid', 'content_wp']]

        pred_dict.update(test_df.set_index('vid')['content_wp'].to_dict())

        print('>>> Random forest MAE on train set: {0:.4f}'.format(mean_absolute_error(train_df.watch_percentage, pred_train_y)))
        print('>>> Random forest MAE on test set: {0:.4f}'.format(mean_absolute_error(test_df.watch_percentage, pred_test_y)))
        print('>>> Random forest R^2 on test set: {0:.4f}'.format(r2_score(test_df.watch_percentage, pred_test_y)))
        print('='*79)
        print()

    # write to txt file
    print('>>> Number of videos in final test result dict: {0}'.format(len(pred_dict)))
    pickle.dump(pred_dict, open('predict_results/content_result-100.p', 'wb'))
