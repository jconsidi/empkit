# empkit/empkit/interp.py

import numpy as np
import sklearn.base

class InterpRegressor(sklearn.base.RegressorMixin, sklearn.base.BaseEstimator):
    """This class is a thin wrapper around np.interp that handles
    duplicates for regression."""
    
    def fit(self, X, y):
        X = np.asarray(X)
        assert len(X.shape) == 2
        assert X.shape[1] == 1

        # first column is projected values, second column is y values
        temp = np.zeros((X.shape[0], 2))
        temp[:, 0] = X[:,0]
        temp[:, 1] = y

        # sort by first column
        temp = temp[temp[:, 0].argsort()]

        write_pos = 0
        group_size = 1

        for read_pos in range(1, X.shape[0]):
            if temp[read_pos, 0] == temp[write_pos, 0]:
                # same x value as previous row
                group_size += 1
                temp[write_pos, 1] += (temp[read_pos, 1] - temp[write_pos, 1]) / group_size
            else:
                # different x value
                group_size = 1
                write_pos += 1
                temp[write_pos,:] = temp[read_pos,:]

        temp = temp[:write_pos+1,:]
        self.xp = temp[:,0].copy()
        self.fp = temp[:,1].copy()

        return self

    def predict(self, X):
        X = np.asarray(X)
        assert len(X.shape) == 2
        assert X.shape[1] == 1

        return np.interp(X[:,0], self.xp, self.fp)
