from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dico.data import FWD, LABELS, NEG, REV, add_reverse_index
from dico.decode import SAFE_DECODERS, decode, joint, self_swap, swap_texts


def random_probs(rng, n):
    x = rng.random((n, len(LABELS))) + 1e-3
    return x / x.sum(axis=1, keepdims=True)


def pair_frame():
    return pd.DataFrame({
        "instance_id": ["p__o", "p__f", "q__o"],
        "pair_id": ["p", "p", "q"],
        "text1_lang": ["en", "en", "en"],
        "text2_lang": ["es", "en", "en"],
        "text1": ["A", "B", "C"],
        "text2": ["B", "A", "D"],
    })


class SelfSwapTest(unittest.TestCase):
    def test_swap_texts_swaps_languages_too(self):
        out = swap_texts(pair_frame())
        self.assertEqual(out.text1.tolist(), ["B", "A", "D"])
        self.assertEqual(out.text1_lang.tolist(), ["es", "en", "en"])

    def test_reversal_consistent_for_any_model(self):
        # Simulate a deterministic model f(text1, text2): the twin row's normal input is
        # this row's swapped input, so self_swap must give Rev-related labels.
        rng = np.random.default_rng(0)
        table = {}

        def model(a, b):
            return table.setdefault((a, b), random_probs(rng, 1)[0])

        for _ in range(50):
            table.clear()
            df = pair_frame()
            probs = np.stack([model(a, b) for a, b in zip(df.text1, df.text2)])
            swap = np.stack([model(b, a) for a, b in zip(df.text1, df.text2)])
            labels = self_swap(df, probs, swap)
            self.assertEqual(labels[1], REV[labels[0]])

    def test_equals_joint_when_twin_present(self):
        df = add_reverse_index(pair_frame().iloc[:2])
        rng = np.random.default_rng(1)
        probs = random_probs(rng, 2)
        swap = probs[[1, 0]]  # p(.|B,A) of row 0 is row 1's prediction, and vice versa
        self.assertEqual(self_swap(df, probs, swap), joint(df, probs))

    def test_neg_bias_moves_decision(self):
        df = pair_frame().iloc[:1]
        probs = np.asarray([[0.05, 0.50, 0.05, 0.40]])
        swap = np.asarray([[0.05, 0.05, 0.50, 0.40]])  # agrees: FWD forward = BWD reversed
        self.assertEqual(self_swap(df, probs, swap, 0.0), [FWD])
        self.assertEqual(self_swap(df, probs, swap, 1.0), [NEG])
        self.assertEqual(decode("self_swap", df, probs, {"neg_bias": 1.0}, swap), [NEG])

    def test_decode_requires_swap_probs(self):
        with self.assertRaises(ValueError):
            decode("self_swap", pair_frame(), random_probs(np.random.default_rng(2), 3))

    def test_safe_list(self):
        self.assertIn("self_swap", SAFE_DECODERS)
        self.assertNotIn("joint_structural", SAFE_DECODERS)
        self.assertNotIn("joint", SAFE_DECODERS)


if __name__ == "__main__":
    unittest.main()
