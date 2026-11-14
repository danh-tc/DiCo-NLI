from __future__ import annotations

import unittest

import torch

from dico.data import BWD, EQ, FWD, LABEL2ID, NEG, REV_PERM
from dico.train import bientail_scores


class BiEntailTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.ab, self.ba = torch.randn(5, 3), torch.randn(5, 3)

    def test_cells_are_a_distribution(self):
        p = bientail_scores(self.ab, self.ba, 0).exp().sum(-1)
        torch.testing.assert_close(p, torch.ones(5), atol=1e-5, rtol=0)

    def test_swapping_inputs_reverses_the_label(self):
        s = bientail_scores(self.ab, self.ba, 0)
        swapped = bientail_scores(self.ba, self.ab, 0)
        torch.testing.assert_close(swapped, s[:, REV_PERM])

    def test_table(self):
        hi = torch.tensor([[9.0, -9.0, -9.0]])  # entailment ~ 1
        lo = torch.tensor([[-9.0, 9.0, -9.0]])  # entailment ~ 0
        cases = {(hi, hi): EQ, (hi, lo): FWD, (lo, hi): BWD, (lo, lo): NEG}
        for (a, b), label in cases.items():
            self.assertEqual(int(bientail_scores(a, b, 0).argmax()), LABEL2ID[label])

    def test_entail_index_is_respected(self):
        a = torch.tensor([[-9.0, -9.0, 9.0]])  # class 2 high
        self.assertEqual(int(bientail_scores(a, a, 2).argmax()), LABEL2ID[EQ])
        self.assertEqual(int(bientail_scores(a, a, 0).argmax()), LABEL2ID[NEG])


if __name__ == "__main__":
    unittest.main()
