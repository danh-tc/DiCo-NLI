from __future__ import annotations

import unittest

import torch

from dico.data import REV_PERM
from dico.train import pair_aware_loss


class PairAwareLossTest(unittest.TestCase):
    def test_exact_reverse_has_no_consistency_penalty(self):
        normal = torch.tensor([[4.0, 2.0, -1.0, 0.0], [0.0, -1.0, 3.0, 1.0]])
        reverse = normal[:, REV_PERM]
        labels = torch.tensor([0, 2])
        without = pair_aware_loss(normal, reverse, labels, 0.0, 0.0)
        with_consistency = pair_aware_loss(normal, reverse, labels, 10.0, 0.0)
        torch.testing.assert_close(without, with_consistency)

    def test_inconsistent_reverse_is_penalized(self):
        normal = torch.tensor([[4.0, 2.0, -1.0, 0.0]])
        reverse = normal.clone()
        labels = torch.tensor([0])
        without = pair_aware_loss(normal, reverse, labels, 0.0, 0.0)
        with_consistency = pair_aware_loss(normal, reverse, labels, 1.0, 0.0)
        self.assertGreater(float(with_consistency), float(without))


if __name__ == "__main__":
    unittest.main()
