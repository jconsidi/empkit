# empkit/empkit/base.py

import numpy as np
import pandas as pd
import sklearn.base

class EnsembleOfManyProjections(sklearn.base.RegressorMixin, sklearn.base.BaseEstimator):
    def __init__(self, *, projections=1):
        super().__init__()

        # number of training points
        self.n = None
        # number of input columns
        self.c = None
        # number of projections
        self.p = projections

        # random projections
        self.projection_matrix = None

    def fit(self, X, y):
        X = np.asarray(X)
        assert len(X.shape) == 2

        (self.n, self.c) = X.shape
        self.projection_matrix = np.random.normal(size=(self.c, self.p))

        # TODO : need to handle ties where whole X row is identical

        projected = X @ self.projection_matrix

        self.sorted_indices = projected.argsort(axis=0)
        assert self.sorted_indices.shape == (self.n, self.p)
        self.sorted_projected = np.take_along_axis(projected, self.sorted_indices, axis=0)
        assert self.sorted_projected.shape == (self.n, self.p)

        self.y = np.array(y)
        assert self.y.shape[0] == self.n

        return self

    def predict(self, X):
        X = np.asarray(X)
        assert len(X.shape) == 2
        assert X.shape[1] == self.projection_matrix.shape[0]

        projected = X @ self.projection_matrix

        output = np.zeros(X.shape[0])
        for j in range(self.p):
            sorted_indices = np.searchsorted(self.sorted_projected[:, j], projected[:, j])

            def lookup_y(sorted_index):
                return self.y[self.sorted_indices[sorted_index, j]]

            def blend_y(k):
                z = self.sorted_projected[k, j]

                # find range with all the tied z values

                k_min = k
                while k_min > 0 and self.sorted_projected[k_min-1, j] == z:
                    k_min = k_min - 1
                assert self.sorted_projected[k_min, j] == z

                k_max = k
                while k_max + 1 < self.n and self.sorted_projected[k_max+1, j] == z:
                    k_max = k_max + 1
                assert self.sorted_projected[k_max, j] == z

                return sum(lookup_y(k2) for k2 in range(k_min, k_max+1)) / (k_max - k_min + 1)

            # TODO: rewrite to be vectorized
            for (i, index_ij) in enumerate(sorted_indices):
                if index_ij == 0:
                    # goes before all the projected values
                    output[i] += blend_y(0)
                elif index_ij < self.n:
                    # interpolation

                    before_z = self.sorted_projected[index_ij - 1, j]
                    curr_z = projected[i, j]
                    after_z = self.sorted_projected[index_ij, j]
                    assert before_z < curr_z <= after_z

                    interp_factor = (curr_z - before_z) / (after_z - before_z)

                    before_y = blend_y(index_ij - 1)
                    after_y = blend_y(index_ij)

                    output[i] += interp_factor * (after_y - before_y) + before_y
                else:
                    # goes after all the projected values
                    output[i] += blend_y(-1)

        return output / self.p
