#!/usr/bin/env python3

import torch
import pytest
from curiosity.metrics import MeanReciprocalRank


def test_mrr():
    logits = torch.tensor([1, 2, 0.5, 0, 4, 3]).reshape(1, 1, -1)
    labels = torch.tensor([0, 1, 0, 0, 0, 1]).reshape(1, 1, -1)
    mask = torch.tensor([1]).reshape(1, 1, -1)
    metric = MeanReciprocalRank()
    mrr = metric(logits, labels, mask)

    # predicted order of documents
    # preds: 4, 5, 1, 0, 2, 3
    # True doc idxs: 1, 5
    # +1 is to make first position/index correspond to rank 1
    # Perfect score is 1
    # MRR of true docs: 1 / (2 + 1) + 1 / (1 + 1) = 1 / 3 + 1 / 2
    # relevant ranks:
    assert pytest.approx(1 / 3 + 1 / 2, mrr.item())
