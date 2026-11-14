from __future__ import annotations

import unittest

import numpy as np

from dico.meta_pool import fit_meta_pool, predict_meta_pool


class MetaPoolTest(unittest.TestCase):
    def test_weights_are_on_simplex(self):
        # View 0 is reliable, view 1 is systematically misleading.
        y = np.asarray([0, 1, 2, 3] * 8)
        good = np.full((len(y), 4), -3.0)
        bad = np.full((len(y), 4), -3.0)
        good[np.arange(len(y)), y] = -0.05
        bad[np.arange(len(y)), (y + 1) % 4] = -0.05
        x = np.stack([good, bad], axis=1)
        model = fit_meta_pool(x, y, ["good", "bad"], l2=0.01, calibrate=False)
        self.assertAlmostEqual(float(model.weights.sum()), 1.0, places=7)
        self.assertTrue((model.weights >= 0).all())
        self.assertGreater(model.weights[0], model.weights[1])
        np.testing.assert_array_equal(predict_meta_pool(x, model), y)


if __name__ == "__main__":
    unittest.main()
