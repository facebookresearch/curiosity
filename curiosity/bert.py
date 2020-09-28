#!/usr/bin/env python3
from typing import Dict, Union

import torch
from pytorch_pretrained_bert.modeling import BertModel

from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel


class BertEncoder(torch.nn.Module):
    """
    Adapted from https://github.com/allenai/allennlp/blob/v0.8.5/allennlp/models/bert_for_classification.py
    and https://github.com/allenai/allennlp/blob/master/allennlp/modules/seq2vec_encoders/bert_pooler.py#L14-L67

    I ran into a lot of trouble trying to get this to work more generically and gave up
    to just implement as a manual switch
    """

    def __init__(
        self,
        bert_model: Union[str, BertModel],
        requires_grad: bool = True,
        index: str = "bert",
    ) -> None:
        super().__init__()

        if isinstance(bert_model, str):
            self.bert_model = PretrainedBertModel.load(bert_model)
        else:
            self.bert_model = bert_model

        for param in self.bert_model.parameters():
            param.requires_grad = requires_grad

        self._embedding_dim = self.bert_model.config.hidden_size
        self._index = index

    def forward(self, tokens: Dict[str, torch.LongTensor]) -> torch.Tensor:
        # pylint: disable=arguments-differ
        input_ids = tokens[self._index]
        token_type_ids = tokens[f"{self._index}-type-ids"]
        input_mask = (input_ids != 0).long()
        # transformers lib doesn't like extra dimensions, and TimeDistributed
        # expects a tensor
        # This works since we only need independent encodings of each piece of text
        if input_ids.dim() > 2:
            shape = input_ids.shape
            word_dim = shape[-1]
            reshaped_input_ids = input_ids.view(-1, word_dim)
            reshaped_token_type_ids = token_type_ids.view(-1, word_dim)
            reshaped_input_mask = input_mask.view(-1, word_dim)
            _, reshaped_pooled = self.bert_model(
                input_ids=reshaped_input_ids,
                token_type_ids=reshaped_token_type_ids,
                attention_mask=reshaped_input_mask,
            )
            pooled = reshaped_pooled.view(shape[:-1] + (-1,))
        else:
            _, pooled = self.bert_model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=input_mask,
            )
        # Current mask is wordpiece mask, we want an utterance mask
        # So search for utterances with all masked wordpieces
        utter_mask = (input_mask.sum(dim=-1) != 0).long()

        return pooled, utter_mask

    def get_output_dim(self) -> int:
        return self._embedding_dim
