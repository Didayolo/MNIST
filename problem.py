# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Youtube: Classify popularity of videos'
_target_column_name = 'view_class'
_additional_drops = ['viewcount']
_prediction_label_names = [0, 1, 2, 3, 4, 5]
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorClassifier()

soft_score_matrix = np.array([
    [1, 0.6, 0, 0, 0, 0],
    [0.3, 1, 0.3, 0, 0, 0],
    [0, 0.3, 1, 0.3, 0, 0],
    [0, 0, 0.3, 1, 0.3, 0],
    [0, 0, 0, 0.3, 1, 0.3],
    [0, 0, 0, 0, 0.6, 1],
])

score_types = [
    rw.score_types.SoftAccuracy(
        name='sacc', score_matrix=soft_score_matrix, precision=3),
    rw.score_types.Accuracy(name='acc', precision=3),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=4, test_size=0.2, random_state=57)
    return cv.split(X, y)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    data['published'] = pd.to_datetime(data['published'])
    y_array = data[_target_column_name].values
    X_df = data.drop(_additional_drops + [_target_column_name], axis=1)
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        return X_df[:100], y_array[:100]
    else:
        return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)
