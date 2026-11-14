from __future__ import annotations

import unittest

import numpy as np

from dico.dynamic_gate import fit_dynamic_gate, predict_dynamic_gate, routing_weights


class DynamicGateTest(unittest.TestCase):
    def test_routing_is_sample_wise_and_on_simplex(self):
        # Expert 0 is reliable on confident class 0/1 examples; expert 1 is
        # reliable on confident class 2/3 examples.  Repeat for stable fitting.
        y = np.asarray([0, 1, 2, 3] * 20)
        x = np.full((len(y), 2, 4), -4.0)
        for i, label in enumerate(y):
            good = 0 if label < 2 else 1
            bad = 1 - good
            x[i, good, label] = -0.02
            x[i, bad, (label + 1) % 4] = -0.02
            x[i, bad, label] = -2.0
        model = fit_dynamic_gate(x, y, ["low", "high"], l2=1e-3, anchor=1e-2)
        weights = routing_weights(x, model)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-8)
        self.assertTrue((weights >= 0).all())
        self.assertGreater(weights[y < 2, 0].mean(), weights[y < 2, 1].mean())
        self.assertGreater(weights[y >= 2, 1].mean(), weights[y >= 2, 0].mean())
        np.testing.assert_array_equal(predict_dynamic_gate(x, model), y)


if __name__ == "__main__":
    unittest.main()
