#!/usr/bin/env python3
from typing import Dict
import torch
from overrides import overrides
from allennlp.common.registrable import Registrable
from allennlp.modules.similarity_functions.similarity_function import SimilarityFunction


class FactRanker(torch.nn.Module, Registrable):
    @overrides
    def forward(self, dialog_context: torch.Tensor, fact_repr: torch.Tensor) -> Dict:
        """
        Accept the dialog context, fact representations, and fact labels
        to produce logits/loss for fact ranking

        The output should be a dictionary with keys 'logits' only if labels are None,
        else 'logits' and 'loss'
        """
        raise NotImplementedError


@FactRanker.register("mean_logit_ranker")
class MeanLogitRanker(FactRanker):
    def __init__(self, similarity_function: SimilarityFunction):
        super().__init__()
        self._similarity_function = similarity_function

    @overrides
    def forward(
        self,
        # TODO: pass in the prior message
        # (batch_size, n_turns, dc_dim)
        shifted_dialog_context: torch.Tensor,
        # One representation vector per fact
        # (batch_size, n_turns, n_facts, repr_dim)
        fact_repr: torch.Tensor,
    ) -> Dict:
        # Make the dims work
        shifted_dc_unsqueeze = shifted_dialog_context.unsqueeze(2)
        logits = self._similarity_function(shifted_dc_unsqueeze, fact_repr)
        return logits
