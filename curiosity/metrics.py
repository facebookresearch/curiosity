#!/usr/bin/env python3

from typing import Optional
from overrides import overrides
from allennlp.training.metrics.metric import Metric
import torch
import numpy as np
from curiosity.util import get_logger


log = get_logger(__name__)


@Metric.register('mean_reciprocal_rank')
class MeanReciprocalRank(Metric):
    def __init__(self):
        self._reciprocal_ranks = []

    def __call__(self,
                 logits: torch.Tensor,
                 labels: torch.Tensor,
                 mask: torch.Tensor):
        """
        logits and labels should be the same shape. Labels should be
        an array of 0/1s to indicate if the document is relevant.

        We don't need a mask here since we select nonzero labels and
        masked entries in labels are never equal to 1 (Pedro is pretty sure)
        """
        n_relevent = labels.sum().item()
        if n_relevent == 0:
            # None are relevent, no-op
            return

        preds = logits.argsort(dim=-1, descending=True)
        # nonzeros occur where there are predictions to make
        # (n_nonzero, 3)
        # 3 = dims for batch, turn and fact
        indices = labels.nonzero()

        # TODO: This could be batched, but its a pain
        all_ranks = []
        #import ipdb; ipdb.set_trace()
        for batch_idx, turn_idx, fact_idx in indices:
            # List of predictions, first element is index
            # of top ranked document, second of second-top, etc
            inst_preds = preds[batch_idx, turn_idx]
            rank = (inst_preds == fact_idx).nonzero().reshape(-1)
            all_ranks.append(rank)
        all_ranks = torch.cat(all_ranks)
        # rank starts at zero from torch, += 1 for inversing it

        reciprocal_ranks = 1 / (1 + all_ranks).float()
        self._reciprocal_ranks.extend(reciprocal_ranks.cpu().numpy().tolist())
        return reciprocal_ranks.mean()

    @overrides
    def get_metric(self, reset: bool = False) -> float:
        if len(self._reciprocal_ranks) == 0:
            log.warn('Taking MRR of zero length list')
            mrr = 0.0
        else:
            mrr = np.array(self._reciprocal_ranks).mean()
        if reset:
            self.reset()
        return mrr

    @overrides
    def reset(self):
        self._reciprocal_ranks = []


@Metric.register("multilabel_micro_precision")
class MultilabelMicroF1(Metric):
    """
    For a problem of (batch_size, *, n_classes) that is multilabel, compute the
    precision, recall, F1 and take the average.
    This is the micro average since each prediction
    bumps the weight by one, whereas the macro average would compute accuracy by
    class, then average the classes

    This assumes that each class logit represents a binary cross entropy problem
    """

    def __init__(self) -> None:
        self._precision_correct_count = 0.0
        self._precision_total_count = 0.0
        self._recall_correct_count = 0.0
        self._recall_total_count = 0.0

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ...).
        gold_labels : ``torch.Tensor``, required.
            A tensor of the same shape as ``predictions``.
        mask: ``torch.Tensor``, optional (default = None).
            A tensor of the same shape as ``predictions``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(
            predictions, gold_labels, mask)

        # Some sanity checks.
        if gold_labels.size() != predictions.size():
            raise ValueError(
                f"gold_labels must have shape == predictions.size() but "
                f"found tensor of shape: {gold_labels.size()}"
            )
        if mask is not None and mask.size() != predictions.size():
            raise ValueError(
                f"mask must have shape == predictions.size() but "
                f"found tensor of shape: {mask.size()}"
            )

        if mask is not None:
            # mask out early to zero out preds/labels  to count
            predictions = predictions * mask
            gold_labels = gold_labels * mask

        # Don't care about batch anymore since its micro-averaged
        predictions = predictions.view(-1)
        gold_labels = gold_labels.view(-1)

        # Find when they are equal, then only count places where the
        # model actually made a prediction
        # If (first is result, second is contrib to denom):
        # - gold is zero and pred is zero -> zero, no contrib
        # - gold is one and pred is zero -> zero, no contrib
        # - gold is zero and pred is one -> zero, contrib
        # - gold is one and pred is one -> one, contrib
        precision = predictions.eq(gold_labels).long() * predictions
        self._precision_correct_count += precision.sum().item()
        self._precision_total_count += predictions.sum().item()

        # Find where they are equal, then only count places where the
        # gold label was true
        recall = predictions.eq(gold_labels).long() * gold_labels
        self._recall_correct_count += recall.sum().item()
        self._recall_total_count += gold_labels.sum().item()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        precision = self._precision_correct_count / max(
            self._precision_total_count, 1.0)
        recall = self._recall_correct_count / max(self._recall_total_count, 1.0)
        f1 = 2 * (precision * recall) / max(precision + recall, 1.0)
        if reset:
            self.reset()
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    @overrides
    def reset(self):
        self._precision_correct_count = 0.0
        self._precision_total_count = 0.0
        self._recall_correct_count = 0.0
        self._recall_total_count = 0.0
