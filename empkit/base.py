# empkit/empkit/base.py

import numpy as np
import pandas as pd
import sklearn.base

from sklearn.model_selection import train_test_split

from .interp import InterpRegressor

class EnsembleOfManyProjections(sklearn.base.RegressorMixin, sklearn.base.BaseEstimator):
    def __init__(self, *, estimator=None, projections=1, boost_train_size=0.5):
        super().__init__()

        self.estimator = estimator
        if estimator is None:
            self.estimator = InterpRegressor()

        # number of training points
        self.n = None
        # number of input columns
        self.c = None
        # number of projections
        self.projections = projections

        # boost train subsampling
        self.boost_train_size = boost_train_size

        # random projections
        self.projection_matrix = None

        self.projection_models = None

    def fit(self, X, y):
        X = np.asarray(X)
        assert len(X.shape) == 2

        (self.n, self.c) = X.shape
        self.projection_matrix = np.random.normal(size=(self.c, self.projections))

        # TODO : need to handle ties where whole X row is identical

        projected = X @ self.projection_matrix

        y_residual = y
        self.projection_models = []
        for j in range(self.projections):
            m = sklearn.base.clone(self.estimator)

            (X_train, _, y_train, _) = train_test_split(projected[:, j:j+1], y_residual, train_size=self.boost_train_size)
            m.fit(X_train, y_train)
            self.projection_models.append(m)
            y_residual = y_residual - m.predict(projected[:, j:j+1])

        return self

    def predict(self, X):
        X = np.asarray(X)
        assert len(X.shape) == 2
        assert X.shape[1] == self.projection_matrix.shape[0]

        projected = X @ self.projection_matrix

        output = np.zeros(X.shape[0])
        for j in range(self.projections):
            output += self.projection_models[j].predict(projected[:, j:j+1])

        return output
