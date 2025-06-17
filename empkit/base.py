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
        self.projections = projections

        # random projections
        self.projection_matrix = None

    def fit(self, X, y):
        X = np.asarray(X)
        assert len(X.shape) == 2

        (self.n, self.c) = X.shape
        self.projection_matrix = np.random.normal(size=(self.c, self.projections))

        # TODO : need to handle ties where whole X row is identical

        projected = X @ self.projection_matrix

        self.projected_pairs = []
        for j in range(self.projections):
            # first column is projected values, second column is y values
            temp = np.zeros((self.n, 2))
            temp[:, 0] = projected[:, j]
            temp[:, 1] = y

            # sort by first column
            temp = temp[temp[:, 0].argsort()]

            write_pos = 0
            group_size = 1

            for read_pos in range(1, self.n):
                if temp[read_pos, 0] == temp[write_pos, 0]:
                    # same projected value as previous row
                    group_size += 1
                    temp[write_pos, 1] += (temp[read_pos, 1] - temp[write_pos, 1]) / group_size
                else:
                    # different projected value
                    group_size = 1
                    write_pos += 1
                    temp[write_pos,:] = temp[read_pos,:]

            self.projected_pairs.append(temp[:write_pos+1, :].copy())
            del temp

        # self.sorted_indices = projected.argsort(axis=0)
        # assert self.sorted_indices.shape == (self.n, self.p)
        # self.sorted_projected = np.take_along_axis(projected, self.sorted_indices, axis=0)
        # assert self.sorted_projected.shape == (self.n, self.p)

        self.y = np.array(y)
        assert self.y.shape[0] == self.n

        return self

    def predict(self, X):
        X = np.asarray(X)
        assert len(X.shape) == 2
        assert X.shape[1] == self.projection_matrix.shape[0]

        projected = X @ self.projection_matrix

        output = np.zeros(X.shape[0])
        for j in range(self.projections):
            sorted_indices = np.searchsorted(self.projected_pairs[j][:,0], projected[:, j])

            def lookup_y(sorted_index):
                # return self.y[self.sorted_indices[sorted_index, j]]
                return self.projected_pairs[j][sorted_index, 1]

            # TODO: rewrite to be vectorized
            for (i, index_ij) in enumerate(sorted_indices):
                if index_ij == 0:
                    # goes before all the projected values
                    output[i] += lookup_y(0)
                elif index_ij < len(self.projected_pairs[j]):
                    # interpolation

                    before_z = self.projected_pairs[j][index_ij - 1, 0]
                    curr_z = projected[i, j]
                    after_z = self.projected_pairs[j][index_ij, 0]
                    assert before_z < curr_z <= after_z

                    interp_factor = (curr_z - before_z) / (after_z - before_z)

                    before_y = lookup_y(index_ij - 1)
                    after_y = lookup_y(index_ij)

                    output[i] += interp_factor * (after_y - before_y) + before_y
                else:
                    # goes after all the projected values
                    assert projected[i, j] >= self.projected_pairs[j][-1, 0]
                    output[i] += lookup_y(-1)

        return output / self.projections
