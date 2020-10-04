#!/usr/bin/env python3
from typing import Dict, Optional, Union
import math
import torch
from torch import nn
from allennlp.nn.util import (
    get_text_field_mask, sequence_cross_entropy_with_logits,
    masked_mean
)
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.training.metrics import CategoricalAccuracy, Average
from curiosity.nn import FactRanker
from curiosity.metrics import MeanReciprocalRank, MultilabelMicroF1
from curiosity.reader import DIALOG_ACT_LABELS
from curiosity.bert import BertEncoder


def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    __constants__ = ['inplace']

    def __init__(self, inplace=False):
        super(GeLU, self).__init__()
        self.inplace = inplace

    def forward(self, input_):
        return gelu(input_)


class Clamp(nn.Module):
    def __init__(self, should_clamp: bool = False):
        super(Clamp, self).__init__()
        self._should_clamp = should_clamp

    def forward(self, input_):
        if self._should_clamp:
            return 0.0 * input_
        else:
            return input_


@Model.register('baseline_curiosity_model')
class CuriosityBaselineModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 use_glove: bool,
                 use_bert: bool,
                 bert_trainable: bool,
                 bert_name: str,
                 mention_embedder: TextFieldEmbedder,
                 dialog_context: FeedForward,
                 fact_ranker: FactRanker,
                 dropout_prob: float,
                 sender_emb_size: int,
                 act_emb_size: int,
                 fact_loss_weight: float,
                 fact_pos_weight: float,
                 utter_embedder: TextFieldEmbedder = None,
                 utter_context: Seq2VecEncoder = None,
                 disable_known_entities: bool = False,
                 disable_dialog_acts: bool = False,
                 disable_likes: bool = False,
                 disable_facts: bool = False):
        super().__init__(vocab)
        self._disable_known_entities = disable_known_entities
        self._disable_dialog_acts = disable_dialog_acts
        self._clamp_dialog_acts = Clamp(should_clamp=disable_dialog_acts)
        self._disable_likes = disable_likes
        self._clamp_likes = Clamp(should_clamp=disable_likes)
        self._disable_facts = disable_facts
        self._clamp_facts = Clamp(should_clamp=disable_facts)

        self._fact_loss_weight = fact_loss_weight
        self._fact_pos_weight = fact_pos_weight

        if int(use_glove) + int(use_bert) != 1:
            raise ValueError('Cannot use bert and glove together')

        self._use_glove = use_glove
        self._use_bert = use_bert
        self._bert_trainable = bert_trainable
        self._bert_name = bert_name
        self._utter_embedder = utter_embedder
        self._utter_context = utter_context
        # Bert encoder is embedder + context
        if use_bert:
            # Not trainable for now
            print('Using BERT encoder ...')
            self._bert_encoder = BertEncoder(
                self._bert_name, requires_grad=bert_trainable
            )
            self._dist_utter_context = None
            self._utter_dim = self._bert_encoder.get_output_dim()
        else:
            print('Using LSTM encoder ...')
            self._bert_encoder = None
            self._dist_utter_context = TimeDistributed(self._utter_context)
            self._utter_dim = self._utter_context.get_output_dim()
        self._dialog_context = dialog_context
        self._fact_ranker = fact_ranker
        # Easier to code as cross entropy with two classes
        # Likes are per message, for only assistant messages
        self._like_classifier = nn.Linear(
            self._dialog_context.get_output_dim(), 2
        )
        self._like_accuracy = CategoricalAccuracy()
        self._like_loss_metric = Average()

        # Dialog acts are per message, for all messages
        # This network predicts the dialog act of the current message
        # for both student and teacher
        self._da_classifier = nn.Sequential(
            nn.Linear(
                self._utter_dim + self._dialog_context.get_output_dim(),
                self._dialog_context.get_output_dim()
            ),
            GeLU(),
            nn.Linear(
                self._dialog_context.get_output_dim(),
                vocab.get_vocab_size(DIALOG_ACT_LABELS)
            )
        )
        self._da_bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self._da_f1_metric = MultilabelMicroF1()
        self._da_loss_metric = Average()

        # This network predicts what the next action should be
        # It predicts for user and assistant since there isn't a real
        # reason to restrict that
        self._policy_classifier = nn.Sequential(
            nn.Linear(
                self._dialog_context.get_output_dim(),
                self._dialog_context.get_output_dim()
            ),
            GeLU(),
            nn.Linear(
                self._dialog_context.get_output_dim(),
                vocab.get_vocab_size(DIALOG_ACT_LABELS)
            )
        )
        self._policy_bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self._policy_f1_metric = MultilabelMicroF1()
        self._policy_loss_metric = Average()

        self._fact_mrr = MeanReciprocalRank()
        self._fact_loss_metric = Average()
        self._dropout_prob = dropout_prob
        self._dropout = nn.Dropout(dropout_prob)
        # Fact use is much less prevalant, about 9 times less so, so factor that in
        self._fact_bce_loss = torch.nn.BCEWithLogitsLoss(
            reduction='none',
            pos_weight=torch.Tensor([self._fact_pos_weight])
        )

    def get_metrics(self, reset: bool = False):
        da_metrics = self._da_f1_metric.get_metric(reset=reset)
        policy_metrics = self._policy_f1_metric.get_metric(reset=reset)
        metrics_to_report = {
            'like_accuracy': self._like_accuracy.get_metric(reset=reset),
            'like_loss': self._like_loss_metric.get_metric(reset=reset),
            'fact_mrr': self._fact_mrr.get_metric(reset=reset),
            'fact_loss': self._fact_loss_metric.get_metric(reset=reset),
            'da_loss': self._da_loss_metric.get_metric(reset=reset),
            'da_micro_f1': da_metrics['f1'],
            'da_micro_precision': da_metrics['precision'],
            'da_micro_recall': da_metrics['recall'],
            'policy_micro_f1': policy_metrics['f1'],
            'policy_micro_precision': policy_metrics['precision'],
            'policy_micro_recall': policy_metrics['recall'],
        }
        metrics_to_report['total'] = \
            metrics_to_report['fact_mrr'] + \
            metrics_to_report['policy_micro_f1'] + \
            metrics_to_report['da_micro_f1'] + \
            metrics_to_report['like_accuracy']

        return metrics_to_report

    def forward(self,
                messages: Dict[str, torch.Tensor],
                # (batch_size, n_turns, n_facts, n_words)
                facts: Dict[str, torch.Tensor],
                # (batch_size, n_turns)
                senders: torch.Tensor,
                # (batch_size, n_turns, n_acts)
                dialog_acts: torch.Tensor,
                # (batch_size, n_turns)
                dialog_acts_mask: torch.Tensor,
                # (batch_size, n_entities)
                known_entities: Dict[str, torch.Tensor],
                # (batch_size, 1)
                focus_entity: Dict[str, torch.Tensor],
                # (batch_size, n_turns, n_facts)
                fact_labels: Optional[torch.Tensor] = None,
                # (batch_size, n_turns, 2)
                likes: Optional[torch.Tensor] = None,
                metadata: Optional[Dict] = None):
        output = {}
        # Take care of the easy stuff first

        if self._use_bert:
            # (batch_size, n_turns, n_words, emb_dim)
            context, utter_mask = self._bert_encoder(messages)
            context = self._dropout(context)
        else:
            # (batch_size, n_turns)
            # This is the mask since not all dialogs have same number
            # of turns
            utter_mask = get_text_field_mask(messages)

            # (batch_size, n_turns, n_words)
            # Mask since not all utterances have same number of words
            # Wrapping dim skips over n_messages dim
            text_mask = get_text_field_mask(messages, num_wrapping_dims=1)
            # (batch_size, n_turns, n_words, emb_dim)
            embed = self._dropout(self._utter_embedder(messages))
            # (batch_size, n_turns, hidden_dim)
            context = self._dist_utter_context(embed, text_mask)

        # (batch_size, n_turns, hidden_dim)
        # n_turns = context.shape[1]
        dialog_context = self._dialog_context(context)

        # (batch_size, n_turns, hidden_dim)
        # This assumes dialog_context does not peek into future
        # dialog_context = self._dialog_context(full_context, utter_mask)

        # shift context one right, pad with zeros at front
        # This makes it so that utter_t is paired with context_t-1
        # which is what we want
        # This is useful in a few different places, so compute it here once
        shape = dialog_context.shape
        shifted_context = torch.cat((
            dialog_context.new_zeros([shape[0], 1, shape[2]]),
            dialog_context[:, :-1, :]
        ), dim=1)
        has_loss = False

        if self._disable_dialog_acts:
            da_loss = 0
            policy_loss = 0
        else:
            # Dialog act per utter loss
            has_loss = True
            da_loss = self._compute_da_loss(
                output,
                context, shifted_context, utter_mask,
                dialog_acts, dialog_acts_mask
            )
            # Policy loss
            policy_loss = self._compute_policy_loss(
                output,
                shifted_context, utter_mask,
                dialog_acts, dialog_acts_mask
            )

        if self._disable_facts:
            # If facts are disabled, don't output anything related
            # to them
            fact_loss = 0
        else:
            if self._use_bert:
                # (batch_size, n_turns, n_words, emb_dim)
                fact_repr, fact_mask = self._bert_encoder(facts)
                fact_repr = self._dropout(fact_repr)
                fact_mask[:, ::2] = 0
            else:
                # (batch_size, n_turns, n_facts)
                # Wrapping dim skips over n_messages
                fact_mask = get_text_field_mask(facts, num_wrapping_dims=1)
                # In addition to masking padded facts, also explicitly mask
                # user turns just in case
                fact_mask[:, ::2] = 0

                # (batch_size, n_turns, n_facts, n_words)
                # Wrapping dim skips over n_turns and n_facts
                fact_text_mask = get_text_field_mask(facts, num_wrapping_dims=2)
                # (batch_size, n_turns, n_facts, n_words, emb_dim)
                # Share encoder with utter encoder
                # Again, stupid dimensions
                fact_embed = self._dropout(self._utter_embedder(facts))
                shape = fact_embed.shape
                word_dim = shape[-2]
                emb_dim = shape[-1]
                reshaped_facts = fact_embed.view(-1, word_dim, emb_dim)
                reshaped_fact_text_mask = fact_text_mask.view(-1, word_dim)
                reshaped_fact_repr = self._utter_context(
                    reshaped_facts, reshaped_fact_text_mask
                )
                # No more emb dimension or word/seq dim
                fact_repr = reshaped_fact_repr.view(shape[:-2] + (-1,))

            fact_logits = self._fact_ranker(
                shifted_context,
                fact_repr,
            )
            output['fact_logits'] = fact_logits
            if fact_labels is not None:
                has_loss = True
                fact_loss = self._compute_fact_loss(
                    fact_logits, fact_labels, fact_mask
                )
                self._fact_loss_metric(fact_loss.item())
                self._fact_mrr(fact_logits, fact_labels, mask=fact_mask)
            else:
                fact_loss = 0

        if self._disable_likes:
            like_loss = 0
        else:
            has_loss = True
            # (batch_size, n_turns, 2)
            like_logits = self._like_classifier(dialog_context)
            output['like_logits'] = like_logits

            # There are several masks here to get the loss/metrics correct
            # - utter_mask: mask out positions that do not have an utterance
            # - user_mask: mask out positions that have a user utterances
            #              since their turns are never liked
            # Using new_ones() preserves the type of the tensor
            user_mask = utter_mask.new_ones(utter_mask.shape)

            # Since the user is always even, this masks out user positions
            user_mask[:, ::2] = 0
            final_mask = utter_mask * user_mask
            masked_likes = likes * final_mask
            if likes is not None:
                has_loss = True
                like_loss = sequence_cross_entropy_with_logits(
                    like_logits, masked_likes, final_mask
                )
                self._like_accuracy(like_logits, masked_likes, final_mask)
                self._like_loss_metric(like_loss.item())
            else:
                like_loss = 0

        if has_loss:
            output['loss'] = (
                self._fact_loss_weight * fact_loss
                + like_loss
                + da_loss + policy_loss
            )

        return output

    def _compute_da_loss(self,
                         output_dict,
                         messages: torch.Tensor,
                         shifted_context: torch.Tensor,
                         utter_mask: torch.Tensor,
                         dialog_acts: torch.Tensor,
                         dialog_acts_mask: torch.Tensor):
        """
        Given utterance at turn t, get the context (utter + acts) from t-1,
        the utter_t, and predict the act
        """
        message_w_context = torch.cat((
            messages, shifted_context
        ), dim=-1)

        # (batch_size, n_turns, n_dialog_acts)
        da_logits = self._da_classifier(message_w_context)
        output_dict['da_logits'] = da_logits
        da_unreduced_loss = self._da_bce_loss(da_logits, dialog_acts.float())
        # Note: the last dimension is expanded from singleton to n_dialog_acts
        # Since the mask is at turn level
        # (batch_size, n_turns, n_dialog_acts)
        da_combined_mask = (
            dialog_acts_mask.float().unsqueeze(-1)
            * utter_mask.float().unsqueeze(-1)
        ).expand_as(da_unreduced_loss)
        da_unreduced_loss = da_combined_mask * da_unreduced_loss
        # Mean loss over non-masked inputs, avoid division by zero
        da_loss = da_unreduced_loss.sum() / da_combined_mask.sum().clamp(min=1)
        da_loss_item = da_loss.item()
        self._da_loss_metric(da_loss_item)
        # (batch_size, n_turns, n_dialog_acts)
        da_preds = (torch.sigmoid(da_logits) > .5).long()
        self._da_f1_metric(da_preds, dialog_acts, da_combined_mask.long())
        return da_loss

    def _compute_policy_loss(self,
                         output_dict,
                         shifted_context: torch.Tensor,
                         utter_mask: torch.Tensor,
                         dialog_acts: torch.Tensor,
                         dialog_acts_mask: torch.Tensor):
        """
        Given utterance at turn t, get the context (utter + acts) from t-1,
        the utter_t, and predict the act
        """

        # (batch_size, n_turns, n_dialog_acts)
        policy_logits = self._policy_classifier(shifted_context)
        output_dict['policy_logits'] = policy_logits
        policy_unreduced_loss = self._policy_bce_loss(policy_logits, dialog_acts.float())
        # Note: the last dimension is expanded from singleton to n_dialog_acts
        # Since the mask is at turn level
        # (batch_size, n_turns, n_dialog_acts)
        policy_combined_mask = (
            dialog_acts_mask.float().unsqueeze(-1)
            * utter_mask.float().unsqueeze(-1)
        ).expand_as(policy_unreduced_loss)
        policy_unreduced_loss = policy_combined_mask * policy_unreduced_loss
        # Mean loss over non-masked inputs, avoid division by zero
        policy_loss = policy_unreduced_loss.sum() / policy_combined_mask.sum().clamp(min=1)
        policy_loss_item = policy_loss.item()
        self._policy_loss_metric(policy_loss_item)
        # (batch_size, n_turns, n_dialog_acts)
        policy_preds = (torch.sigmoid(policy_logits) > .5).long()
        self._policy_f1_metric(policy_preds, dialog_acts, policy_combined_mask.long())
        return policy_loss

    def _compute_fact_loss(self, logits: torch.Tensor,
                           fact_labels: torch.Tensor, fact_mask: torch.Tensor):
        # Don't reduce to mask out padded facts
        unreduced_loss = self._fact_bce_loss(logits, fact_labels)
        total_loss = (
            unreduced_loss * fact_mask.float()
        ).sum()
        mean_loss = total_loss / fact_mask.float().sum()
        return mean_loss
