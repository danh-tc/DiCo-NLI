import numpy as np
import pandas as pd
import torch

from dico.data import LABELS
from dico.debias import surface_bias_logprobs
from dico.train import pair_aware_loss


def toy_df(n_pairs: int = 40) -> pd.DataFrame:
    rows = []
    for i in range(n_pairs):
        a, b = f"a black dog {i}", f"a dog {i}"
        rows.append((f"p{i}__en-en__original", f"p{i}", a, b, "FORWARD_ENTAILMENT", 1))
        rows.append((f"p{i}__en-en__flipped", f"p{i}", b, a, "BACKWARD_ENTAILMENT", 1))
        rows.append((f"q{i}__en-en__original", f"q{i}", f"two cats {i}", f"a white cat {i}", "NEGATIVE_OTHER", 1))
        rows.append((f"r{i}__en-en__original", f"r{i}", f"a man {i}", f"a man {i}", "EQUIVALENCE", 1))
    return pd.DataFrame(rows, columns=["instance_id", "pair_id", "text1", "text2", "label", "track"])


def test_bias_logprobs_shape_and_normalised():
    lp, lp_rev, report = surface_bias_logprobs(toy_df())
    assert lp.shape == lp_rev.shape == (160, len(LABELS))
    np.testing.assert_allclose(np.exp(lp).sum(1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.exp(lp_rev).sum(1), 1.0, atol=1e-5)
    assert report["bias_accuracy"] > 0.5  # the toy data is surface-solvable


def test_poe_loss_without_bias_is_unchanged():
    torch.manual_seed(0)
    a, b = torch.randn(6, 4), torch.randn(6, 4)
    y = torch.randint(0, 4, (6,))
    uniform = torch.full((6, 4), float(np.log(0.25)))
    plain = pair_aware_loss(a, b, y, 0.2, 0.0)
    # a uniform bias adds a constant to every logit: the loss must not change
    assert torch.allclose(plain, pair_aware_loss(a, b, y, 0.2, 0.0, uniform, uniform), atol=1e-5)


def test_poe_confident_bias_shrinks_gradient():
    torch.manual_seed(0)
    a = torch.zeros(4, 4, requires_grad=True)
    b = torch.zeros(4, 4)
    y = torch.zeros(4, dtype=torch.long)
    confident = torch.log(torch.tensor([[0.97, 0.01, 0.01, 0.01]]).repeat(4, 1))
    plain = pair_aware_loss(a, b, y, 0.0, 0.0)
    (g_plain,) = torch.autograd.grad(plain, a)
    poe = pair_aware_loss(a, b, y, 0.0, 0.0, confident, confident)
    (g_poe,) = torch.autograd.grad(poe, a)
    assert g_poe.abs().sum() < g_plain.abs().sum()


def test_diff_segment():
    from dico.train import diff_segment

    assert diff_segment("Two cats", "A white cat") == "A white cat | only in first: two, cats | only in second: a, white, cat"
    assert diff_segment("a dog", "a dog").endswith("only in first: - | only in second: -")
