#!/usr/bin/env python3

import unittest

from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_regression

from empkit import EnsembleOfManyProjections

class EMPCVTestCase(unittest.TestCase):
    def test_00_cv(self):
        X, y = make_regression(n_samples=100, n_features=10, noise=0.1)

        emp = EnsembleOfManyProjections(projections=5)

        scores = cross_val_score(emp, X, y, cv=5, scoring="r2")

        assert len(scores) == 5
