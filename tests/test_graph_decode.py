from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dico.data import BWD, EQ, FWD, LABELS, NEG
from dico.graph_decode import graph_map, temperature_scale


def prob(**values) -> list[float]:
    base = {label: 0.01 for label in LABELS}
    base.update(values)
    total = sum(base.values())
    return [base[label] / total for label in LABELS]


class GraphDecodeTest(unittest.TestCase):
    def test_hard_reversal_uses_joint_map(self):
        df = pd.DataFrame({
            "pair_id": ["p", "p"],
            "text1_lang": ["en", "en"],
            "text2_lang": ["en", "en"],
            "text1": ["A", "B"],
            "text2": ["B", "A"],
            "rev_idx": [1, 0],
        })
        probs = np.asarray([
            prob(**{FWD: 0.92, NEG: 0.04, BWD: 0.02}),
            prob(**{NEG: 0.52, BWD: 0.44, FWD: 0.02}),
        ])
        self.assertEqual(graph_map(df, probs), [FWD, BWD])

    def test_soft_transitivity_fixes_low_confidence_shortcut(self):
        df = pd.DataFrame({
            "pair_id": ["ab", "bc", "ac"],
            "text1_lang": ["en"] * 3,
            "text2_lang": ["en"] * 3,
            "text1": ["A", "B", "A"],
            "text2": ["B", "C", "C"],
            "rev_idx": [-1, -1, -1],
        })
        probs = np.asarray([
            prob(**{FWD: 0.94, EQ: 0.02, BWD: 0.02}),
            prob(**{FWD: 0.94, EQ: 0.02, BWD: 0.02}),
            prob(**{NEG: 0.54, FWD: 0.42, EQ: 0.02}),
        ])
        labels, stats = graph_map(
            df, probs, transitivity_penalty=1.0, return_stats=True
        )
        self.assertEqual(labels, [FWD, FWD, FWD])
        self.assertEqual(stats.observed_paths, 1)

    def test_undefined_composition_is_not_forced(self):
        df = pd.DataFrame({
            "pair_id": ["ab", "bc", "ac"],
            "text1_lang": ["en"] * 3,
            "text2_lang": ["en"] * 3,
            "text1": ["A", "B", "A"],
            "text2": ["B", "C", "C"],
            "rev_idx": [-1, -1, -1],
        })
        probs = np.asarray([
            prob(**{FWD: 0.96}),
            prob(**{BWD: 0.96}),
            prob(**{NEG: 0.96}),
        ])
        self.assertEqual(graph_map(df, probs, transitivity_penalty=None), [FWD, BWD, NEG])

    def test_temperature_preserves_row_argmax(self):
        probs = np.asarray([[0.7, 0.1, 0.1, 0.1], [0.1, 0.2, 0.6, 0.1]])
        scaled = temperature_scale(probs, 2.5)
        np.testing.assert_array_equal(scaled.argmax(1), probs.argmax(1))
        np.testing.assert_allclose(scaled.sum(1), 1.0)


if __name__ == "__main__":
    unittest.main()
