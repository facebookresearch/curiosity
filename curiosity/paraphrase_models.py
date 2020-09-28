#!/usr/bin/env python3
from typing import Dict, List, Tuple, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell
from torch.nn import EmbeddingBag, Sequential

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.attention import LegacyAttention
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.feedforward import FeedForward
from allennlp.nn import util, RegularizerApplicator
from allennlp.nn.beam_search import BeamSearch
from allennlp.training.metrics import BLEU


@Model.register("curiosity_paraphrase_seq2seq")
class FactParaphraseSeq2Seq(Model):
    """
    Given facts and dialog acts, it generates the paraphrased message.
    TODO: add dialog & dialog acts history

    This implementation is based off the default SimpleSeq2Seq model,
    which takes a sequence, encodes it, and then uses the encoded
    representations to decode another sequence.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        source_embedder: TextFieldEmbedder,
        source_encoder: Seq2SeqEncoder,
        max_decoding_steps: int,
        dialog_acts_encoder: FeedForward = None,
        attention: Attention = None,
        attention_function: SimilarityFunction = None,
        n_dialog_acts: int = None,
        beam_size: int = None,
        target_namespace: str = "tokens",
        target_embedding_dim: int = None,
        scheduled_sampling_ratio: float = 0.0,
        use_bleu: bool = True,
        use_dialog_acts: bool = True,
        regularizers: Optional[RegularizerApplicator] = None
    ) -> None:
        super().__init__(vocab, regularizers)
        self._target_namespace = target_namespace
        self._scheduled_sampling_ratio = scheduled_sampling_ratio

        # We need the start symbol to provide as the input at the first
        # timestep of decoding, and end symbol as a way to indicate the end
        # of the decoded sequence.
        self._start_index = \
            self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = \
            self.vocab.get_token_index(END_SYMBOL, self._target_namespace)

        if use_bleu:
            pad_index = self.vocab.get_token_index(
                self.vocab._padding_token, self._target_namespace
            )
            self._bleu = \
                BLEU(exclude_indices={pad_index, self._end_index, self._start_index})
        else:
            self._bleu = None

        # At prediction time, we use a beam search to find the most
        # likely sequence of target tokens.
        beam_size = beam_size or 1
        self._max_decoding_steps = max_decoding_steps
        self._beam_search = BeamSearch(
            self._end_index, max_steps=max_decoding_steps, beam_size=beam_size
        )

        # Dense embedding of source (Facts) vocab tokens.
        self._source_embedder = source_embedder

        # Encodes the sequence of source embeddings into a sequence of hidden states.
        self._source_encoder = source_encoder

        if use_dialog_acts:
            # Dense embedding of dialog acts.
            da_embedding_dim = dialog_acts_encoder.get_input_dim()
            self._dialog_acts_embedder = EmbeddingBag(n_dialog_acts, da_embedding_dim)

            # Encodes dialog acts
            self._dialog_acts_encoder = dialog_acts_encoder

        else:
            self._dialog_acts_embedder = None
            self._dialog_acts_encoder = None

        num_classes = self.vocab.get_vocab_size(self._target_namespace)

        # Attention mechanism applied to the encoder output for each step.
        if attention:
            if attention_function:
                raise ConfigurationError(
                    "You can only specify an attention module or an "
                    "attention function, but not both."
                )
            self._attention = attention
        elif attention_function:
            self._attention = LegacyAttention(attention_function)
        else:
            self._attention = None

        # Dense embedding of vocab words in the target space.
        target_embedding_dim = target_embedding_dim or source_embedder.get_output_dim()
        self._target_embedder = Embedding(num_classes, target_embedding_dim)

        # Decoder output dim needs to be the same as the encoder output dim
        # since we initialize the hidden state of the decoder with the final
        # hidden state of the encoder.
        self._encoder_output_dim = self._source_encoder.get_output_dim()
        if use_dialog_acts:
            self._merge_encoder = Sequential(
                Linear(
                    self._source_encoder.get_output_dim()
                    + self._dialog_acts_encoder.get_output_dim(),
                    self._encoder_output_dim
                )
            )
        self._decoder_output_dim = self._encoder_output_dim

        if self._attention:
            # If using attention, a weighted average over encoder outputs will
            # be concatenated to the previous target embedding to form the input
            # to the decoder at each time step.
            self._decoder_input_dim = self._decoder_output_dim + target_embedding_dim
        else:
            # Otherwise, the input to the decoder is just the previous target embedding.
            self._decoder_input_dim = target_embedding_dim

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        # TODO (pradeep): Do not hardcode decoder cell type.
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = Linear(self._decoder_output_dim, num_classes)

    def take_step(
        self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.
        """
        # shape: (group_size, num_classes)
        output_projections, state = \
            self._prepare_output_projections(last_predictions, state)

        # shape: (group_size, num_classes)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)

        return class_log_probabilities, state

    @overrides
    def forward(
        self,  # type: ignore
        source_tokens: Dict[str, torch.LongTensor],
        target_tokens: Dict[str, torch.LongTensor] = None,
        dialog_acts: Optional[torch.Tensor] = None,
        sender: Optional[torch.Tensor] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Make foward pass with decoder logic for producing the entire target sequence.
        """
        source_state, dialog_acts_state = self._encode(source_tokens, dialog_acts)

        if target_tokens:
            state = self._init_decoder_state(source_state, dialog_acts_state)
            # The `_forward_loop` decodes the input sequence and
            # computes the loss during training and validation.
            output_dict = self._forward_loop(state, target_tokens)
        else:
            output_dict = {}

        if not self.training:
            state = self._init_decoder_state(source_state, dialog_acts_state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)
            if target_tokens and self._bleu:
                # shape: (batch_size, beam_size, max_sequence_length)
                top_k_predictions = output_dict["predictions"]
                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]
                self._bleu(best_predictions, target_tokens["tokens"])

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            # Beam search gives us the top k results for each source sentence
            # in the batch but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[: indices.index(self._end_index)]
            predicted_tokens = [
                self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                for x in indices
            ]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def _encode(
        self,
        source_tokens: Dict[str, torch.Tensor],
        dialog_acts: torch.Tensor = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # Encode source tokens
        source_state = self._encode_source_tokens(source_tokens)

        # Encode dialog acts
        if self._dialog_acts_encoder:
            dialog_acts_state = self._encode_dialog_acts(dialog_acts)

        else:
            dialog_acts_state = None

        return (source_state, dialog_acts_state)

    def _encode_source_tokens(
        self,
        source_tokens: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._source_encoder(embedded_input, source_mask)
        return {"source_mask": source_mask, "encoder_outputs": encoder_outputs}

    def _encode_dialog_acts(self, dialog_acts: torch.Tensor) -> torch.Tensor:
        # shape: (batch_size, dialog_acts_embeddings_size)
        embedded_dialog_acts = self._dialog_acts_embedder(dialog_acts)

        # shape: (batch_size, dim_encoder)
        dialog_acts_state = self._dialog_acts_encoder(embedded_dialog_acts)
        return dialog_acts_state

    def _init_decoder_state(
        self,
        source_state: Dict[str, torch.Tensor],
        dialog_acts_state: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        batch_size = source_state["source_mask"].size(0)

        # shape: (batch_size, encoder_output_dim)
        final_encoder_output = util.get_final_encoder_states(
            source_state["encoder_outputs"],
            source_state["source_mask"],
            self._source_encoder.is_bidirectional()
        )

        # Condition the source tokens state with dialog acts state
        if self._dialog_acts_encoder:
            final_encoder_output = self._merge_encoder(
                torch.cat([final_encoder_output, dialog_acts_state], dim=1))

        # Initialize the decoder hidden state with the final output of the encoder.
        # shape: (batch_size, decoder_output_dim)
        source_state["decoder_hidden"] = final_encoder_output
        # shape: (batch_size, decoder_output_dim)
        source_state["decoder_context"] = source_state["encoder_outputs"].new_zeros(
            batch_size, self._decoder_output_dim
        )
        return source_state

    def _forward_loop(
        self,
        state: Dict[str, torch.Tensor],
        target_tokens: Dict[str, torch.LongTensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Make forward pass during training or do greedy search during prediction.
        Notes
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.
        """
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        batch_size = source_mask.size()[0]

        if target_tokens:
            # shape: (batch_size, max_target_sequence_length)
            targets = target_tokens["tokens"]

            _, target_sequence_length = targets.size()

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps

        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        last_predictions = \
            source_mask.new_full((batch_size,), fill_value=self._start_index)

        step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                # Use gold tokens at test time and at a rate of
                # 1 - _scheduled_sampling_ratio during training.
                # shape: (batch_size,)
                input_choices = last_predictions
            elif not target_tokens:
                # shape: (batch_size,)
                input_choices = last_predictions
            else:
                # shape: (batch_size,)
                input_choices = targets[:, timestep]

            # shape: (batch_size, num_classes)
            output_projections, state = \
                self._prepare_output_projections(input_choices, state)

            # list of tensors, shape: (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))

            # shape: (batch_size, num_classes)
            class_probabilities = F.softmax(output_projections, dim=-1)

            # shape (predicted_classes): (batch_size,)
            _, predicted_classes = torch.max(class_probabilities, 1)

            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes

            step_predictions.append(last_predictions.unsqueeze(1))

        # shape: (batch_size, num_decoding_steps)
        predictions = torch.cat(step_predictions, 1)

        output_dict = {"predictions": predictions}

        if target_tokens:
            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)

            # Compute loss.
            target_mask = util.get_text_field_mask(target_tokens)
            loss = self._get_loss(logits, targets, target_mask)
            output_dict["loss"] = loss

        return output_dict

    def _forward_beam_search(
        self,
        state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Make forward pass during prediction using a beam search."""
        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full(
            (batch_size,), fill_value=self._start_index
        )

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_predictions, state, self.take_step
        )

        output_dict = {
            "class_log_probabilities": log_probabilities,
            "predictions": all_top_k_predictions,
        }
        return output_dict

    def _prepare_output_projections(
        self,
        last_predictions: torch.Tensor,
        state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.
        Inputs are the same as for `take_step()`.
        """
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (group_size, decoder_output_dim)
        decoder_hidden = state["decoder_hidden"]

        # shape: (group_size, decoder_output_dim)
        decoder_context = state["decoder_context"]

        # shape: (group_size, target_embedding_dim)
        embedded_input = self._target_embedder(last_predictions)

        if self._attention:
            # shape: (group_size, encoder_output_dim)
            attended_input = self._prepare_attended_input(
                decoder_hidden, encoder_outputs, source_mask
            )

            # shape: (group_size, decoder_output_dim + target_embedding_dim)
            decoder_input = torch.cat((attended_input, embedded_input), -1)
        else:
            # shape: (group_size, target_embedding_dim)
            decoder_input = embedded_input

        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        decoder_hidden, decoder_context = self._decoder_cell(
            decoder_input, (decoder_hidden, decoder_context)
        )

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context

        # shape: (group_size, num_classes)
        output_projections = self._output_projection_layer(decoder_hidden)

        return output_projections, state

    def _prepare_attended_input(
        self,
        decoder_hidden_state: torch.LongTensor = None,
        encoder_outputs: torch.LongTensor = None,
        encoder_outputs_mask: torch.LongTensor = None,
    ) -> torch.Tensor:
        """Apply attention over encoder outputs and decoder state."""
        # Ensure mask is also a FloatTensor. Or else the multiplication within
        # attention will complain.
        # shape: (batch_size, max_input_sequence_length)
        encoder_outputs_mask = encoder_outputs_mask.float()

        # shape: (batch_size, max_input_sequence_length)
        input_weights = \
            self._attention(decoder_hidden_state, encoder_outputs, encoder_outputs_mask)

        # shape: (batch_size, encoder_output_dim)
        attended_input = util.weighted_sum(encoder_outputs, input_weights)

        return attended_input

    @staticmethod
    def _get_loss(
        logits: torch.LongTensor,
        targets: torch.LongTensor,
        target_mask: torch.LongTensor
    ) -> torch.Tensor:

        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._bleu and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        return all_metrics
