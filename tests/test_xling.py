from __future__ import annotations

import unittest

import pandas as pd
import torch

from dico.data import REV_PERM
from dico.train import PairAwareCollator, cross_lingual_partners, xling_loss


def frame():
    # One source pair p: en-en and es-es in both orientations, plus one mixed row.
    return pd.DataFrame({
        "instance_id": ["p__en-en__original", "p__en-en__flipped", "p__es-es__original",
                        "p__es-es__flipped", "p__en-es__flipped", "q__en-en__original"],
        "pair_id": ["p", "p", "p", "p", "p", "q"],
        "text1_lang": ["en", "en", "es", "es", "en", "en"],
        "text2_lang": ["en", "en", "es", "es", "es", "en"],
    })


class PartnersTest(unittest.TestCase):
    def test_js_partners_exclude_same_language_and_mark_orientation(self):
        partners = cross_lingual_partners(frame(), "js")
        self.assertEqual(partners[0], [(2, False), (3, True), (4, True)])
        self.assertEqual(partners[5], [])  # no other language view of q

    def test_en_teacher_only(self):
        partners = cross_lingual_partners(frame(), "en")
        self.assertEqual(partners[0], [])  # en-en rows are the teacher
        self.assertEqual(partners[3], [(0, True), (1, False)])
        self.assertEqual(partners[4], [(0, True), (1, False)])

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            cross_lingual_partners(frame(), "nope")


class XlingLossTest(unittest.TestCase):
    def test_zero_for_aligned_views(self):
        normal = torch.tensor([[3.0, 1.0, -1.0, 0.0], [0.0, 2.0, -1.0, 1.0]])
        partner = torch.stack([normal[0], normal[1][REV_PERM]])  # second partner is reversed
        flip = torch.tensor([False, True])
        mask = torch.tensor([True, True])
        for mode in ("js", "en"):
            self.assertAlmostEqual(float(xling_loss(normal, partner, flip, mask, mode)), 0.0, places=6)

    def test_positive_when_disagreeing_and_masked_rows_ignored(self):
        normal = torch.tensor([[3.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]])
        partner = torch.tensor([[0.0, 3.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]])
        flip = torch.tensor([False, False])
        self.assertGreater(float(xling_loss(normal, partner, flip, torch.tensor([True, False]), "js")), 0)
        self.assertEqual(float(xling_loss(normal, partner, flip, torch.tensor([False, False]), "js")), 0)

    def test_en_teacher_gets_no_gradient(self):
        normal = torch.zeros(1, 4, requires_grad=True)
        partner = torch.tensor([[2.0, 0.0, 0.0, 0.0]], requires_grad=True)
        xling_loss(normal, partner, torch.tensor([False]), torch.tensor([True]), "en").backward()
        self.assertIsNone(partner.grad)
        self.assertIsNotNone(normal.grad)


class CollatorTest(unittest.TestCase):
    def test_three_views_share_padding(self):
        class Tok:  # minimal stand-in for DataCollatorWithPadding
            pad_token_id = 0
            padding_side = "right"
            model_input_names = ["input_ids", "attention_mask"]

            def pad(self, features, **kwargs):
                width = max(len(f["input_ids"]) for f in features)
                ids = [f["input_ids"] + [0] * (width - len(f["input_ids"])) for f in features]
                mask = [[1] * len(f["input_ids"]) + [0] * (width - len(f["input_ids"])) for f in features]
                return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}

        rows = [
            {"input_ids": [1, 2], "rev_input_ids": [2, 1], "par_input_ids": [5, 6, 7],
             "par_flip": 1, "par_mask": 1, "labels": 0},
            {"input_ids": [3], "rev_input_ids": [3], "par_input_ids": [8],
             "par_flip": 0, "par_mask": 0, "labels": 2},
        ]
        collator = PairAwareCollator.__new__(PairAwareCollator)
        collator.base = lambda feats: Tok().pad(feats)
        batch = collator(rows)
        self.assertEqual(batch["input_ids"].shape, batch["par_input_ids"].shape)
        self.assertEqual(batch["rev_input_ids"].shape[1], 3)
        self.assertEqual(batch["par_flip"].tolist(), [True, False])
        self.assertEqual(batch["labels"].tolist(), [0, 2])


if __name__ == "__main__":
    unittest.main()
