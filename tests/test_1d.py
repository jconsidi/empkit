#!/usr/bin/env python3

import unittest

from empkit import EnsembleOfManyProjections

class EMP1dTestCase(unittest.TestCase):
    def check_prediction(self, m, X, y):
        with self.subTest(X=X):
            self.check_value(m.predict([X]), y)

    def check_value(self, v, v_expected):
        self.assertEqual(v.size, 1)
        self.assertAlmostEqual(v.reshape(()), v_expected)
    
    def test_line(self):
        for p in range(1, 21):
            with self.subTest(projections=p):
                m = EnsembleOfManyProjections(projections=p)
                m.fit([[0], [1]], [0, 1])

                # training data points
                self.check_prediction(m, [0.00], 0.00)
                self.check_prediction(m, [1.00], 1.00)

                # interpolation
                self.check_prediction(m, [0.25], 0.25)
                self.check_prediction(m, [0.50], 0.50)
                self.check_prediction(m, [0.75], 0.75)

                # extrapolation
                self.check_prediction(m, [-1.00], 0.00)
                self.check_prediction(m, [-0.75], 0.00)
                self.check_prediction(m, [-0.50], 0.00)
                self.check_prediction(m, [-0.25], 0.00)
                self.check_prediction(m, [+1.25], 1.00)
                self.check_prediction(m, [+1.50], 1.00)
                self.check_prediction(m, [+1.75], 1.00)
                self.check_prediction(m, [+2.00], 1.00)
