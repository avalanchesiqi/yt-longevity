#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Ridge regressor."""

from __future__ import division, print_function
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge


class RidgeRegressor:
    """A Ridge Regressor that takes train and test data as input, output predict value.
    
    Attributes:
        train_x: training features matrix
        train_y: training label vector
        test_x: test features matrix
        test_y: test label vector
        cv_ratio: ratio of cv data over all training data, default at 0.2
    """

    def __init__(self, train_x, train_y, test_x, test_y, cv_ratio=0.2):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.cv_ratio = cv_ratio

    @classmethod
    def from_dense(cls, train, test, cv_ratio=0.2):
        """Build class from dense matrix"""
        train_x = train[:, :-1]
        train_y = train[:, -1]
        test_x = test[:, :-1]
        test_y = test[:, -1]
        return cls(train_x, train_y, test_x, test_y, cv_ratio)

    def predict(self):
        """Predict test dataset with search best alpha value in train/cv dataset"""
        train_x, cv_x, train_y, cv_y = train_test_split(self.train_x, self.train_y,
                                                        train_size=1 - self.cv_ratio, test_size=self.cv_ratio)

        verbose = True
        if verbose:
            print('>>> Shape of train matrix: {0} x {1}'.format(*self.train_x.shape))
            print('>>> Shape of test matrix: {0} x {1}'.format(*self.test_x.shape))

        # grid search best alpha value over -5 to 5 in log space
        alpha_array = [10 ** t for t in range(-5, 5)]
        cv_mae = []
        for alpha in alpha_array:
            predictor = Ridge(alpha=alpha)
            predictor.fit(train_x, train_y)
            cv_yhat = predictor.predict(cv_x)
            mae = mean_absolute_error(cv_y, cv_yhat)
            print('>>> MAE at alpha {0}: {1:.4f}'.format(alpha, mae))
            cv_mae.append(mae)

        # build the best predictor
        best_alpha_idx = np.argmin(np.array(cv_mae))
        best_alpha = alpha_array[best_alpha_idx]
        print('>>> Best hyper parameter alpha: {0}'.format(best_alpha))
        best_predictor = Ridge(alpha=best_alpha)
        best_predictor.fit(self.train_x, self.train_y)

        # predict test dataset
        test_yhat = best_predictor.predict(self.test_x)
        print('>>> Predict {0} videos in test dataset'.format(len(test_yhat)))
        print('>>> Ridge model: MAE of test dataset: {0}'.format(mean_absolute_error(self.test_y, test_yhat)))
        print('>>> Ridge model: R2 of test dataset: {0}'.format(r2_score(self.test_y, test_yhat)))
        return test_yhat
