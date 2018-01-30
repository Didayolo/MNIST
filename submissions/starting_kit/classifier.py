# -*- coding: utf-8 -*-

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.base import BaseEstimator


def lmap(f, l):
    return list(map(f, l))


def amap(f, l):
    return np.array(lmap(f, l))


class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = ExtraTreesClassifier(max_depth=10, random_state=42)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        y_pred = self.predict(X)
        return amap(lambda tr: [1 if i == tr else 0 for i in range(6)], y_pred)
