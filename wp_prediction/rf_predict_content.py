from __future__ import division, print_function
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


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

    # do the first column
    cat_observations = observations[:, 1]

    # do the others
    for i in range(2, observations.shape[1]):
        enc_label = LabelEncoder()
        cat_observations = np.column_stack((cat_observations, enc_label.fit_transform(observations[:, i])))
        print(enc_label.classes_)

    cat_observations_values = cat_observations.astype(float)

    enc_onehot = OneHotEncoder()
    cat_observations_data = enc_onehot.fit_transform(cat_observations_values)
    cat_observations_data = cat_observations_data.toarray()

    # add control variable
    cat_observations_data = np.column_stack((cat_observations_data, observations[:, 0]))

    train_x = cat_observations_data[:num_train, :]
    test_x = labels[:num_train]
    train_y = cat_observations_data[num_train:, :]
    test_y = labels[num_train:]

    return train_x, test_x, train_y, test_y


def remove_invalid_prediction(arr):
    arr[arr > 1] = 1
    arr[arr < 0] = 0
    return arr


if __name__ == '__main__':
    category = 'sports'
    train_loc = '../../data/production_data/random_langs/train_data/{0}.txt'.format(category)
    test_loc = '../../data/production_data/random_langs/test_data/{0}.txt'.format(category)

    print('>>> Train on category: {0}'.format(category))

    train_x, train_y, test_x, test_y = build_matrix(train_loc, test_loc)

    rf_regressor = RandomForestRegressor(n_estimators=100, min_samples_leaf=50, random_state=42)
    rf_regressor.fit(train_x, train_y)
    print(rf_regressor)
    print(rf_regressor.feature_importances_)
    print('>>> Number of features: {0}'.format(rf_regressor.n_features_))

    pred_train_y = rf_regressor.predict(train_x)
    pred_train_y = remove_invalid_prediction(pred_train_y)

    pred_test_y = rf_regressor.predict(test_x)
    pred_test_y = remove_invalid_prediction(pred_test_y)

    print('>>> Random forest MAE on train set: {0:.4f}'.format(mean_absolute_error(train_y, pred_train_y)))
    print('>>> Random forest MAE on test set: {0:.4f}'.format(mean_absolute_error(test_y, pred_test_y)))
    print('>>> Random forest R^2 on test set: {0:.4f}'.format(rf_regressor.score(test_x, test_y)))

