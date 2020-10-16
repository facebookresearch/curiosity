#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.

"""
Reader for curiosity dialog dataset. Below is a sample json with relevant structure


{
    "dialogs": [
        {
            "messages": [
                {
                    "message": "Hi, what do you know about St. Louis' history?",
                    "liked": false,
                    "sender": "user",
                    "facts": []
                },
                {
                    "message": "St. Louis had among worst air pollution in U.S.?",
                    "liked": true,
                    "sender": "assistant",
                    "facts": [
                        {
                        "fid": 54538,
                        "used": true
                        },
                        {
                        "fid": 54472,
                        "used": false
                        },
                        {
                        "fid": 54490,
                        "used": false
                        },
                        {
                        "fid": 54701,
                        "used": false
                        },
                        {
                        "fid": 54646,
                        "used": false
                        },
                        {
                        "fid": 54681,
                        "used": false
                        },
                        {
                        "fid": 54746,
                        "used": false
                        },
                        {
                        "fid": 54523,
                        "used": false
                        },
                        {
                        "fid": 54526,
                        "used": false
                        }
                    ]
                },
            ],
            "known_entities": [
                "Major League Baseball",
                "United Kingdom",
                "United States",
                "United States Census Bureau",
                "Missouri River"
            ],
            "focus_entity": "St. Louis",
            "dialog_id": 77,
            "inferred_steps": false,
            "created_time": 1568060716,
            "aspects": [
                "History",
                "Education"
            ],
            "first_aspect": "History",
            "second_aspect": "Education",
            "shuffle_facts": true,
            "related_entities": [
                "Auguste Chouteau",
                "Spain",
                "Susan Polgar",
                "Darby, Pennsylvania",
                "MacArthur Bridge (St. Louis)",
                "Christ Church Cathedral, Oxford",
                "Mound City, Illinois",
                "Major League Baseball",
                "United Kingdom",
                "United States",
                "Washington University in St. Louis",
                "United States Census Bureau",
                "Greater St. Louis",
                "Missouri River"
            ]
        }
    ]
}
"""
from typing import Dict, Optional, List
import json
import csv
import os

import numpy as np
from overrides import overrides
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.fields import (
    TextField,
    ListField,
    MetadataField,
    LabelField,
    ArrayField,
    MultiLabelField,
)
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter

from curiosity.db import verify_checksum, create_sql, Fact


USER = "user"
ASSISTANT = "assistant"
DIALOG_ACT_LABELS = "dialog_act_labels"


class MultiLabelFieldListCompat(MultiLabelField):
    """
    Fixes a bug where if the field is used in a ListField, that the
    number of labels is lost and causes an error.
    """

    @overrides
    def empty_field(self):
        return MultiLabelField(
            [], self._label_namespace, skip_indexing=True, num_labels=self._num_labels
        )


def to_long_field(nums: List[int]) -> ArrayField:
    return ArrayField(np.array(nums, dtype=np.long), dtype=np.long)


@DatasetReader.register("curiosity_dialog")
class CuriosityDialogReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        mention_tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        mention_indexers: Dict[str, TokenIndexer] = None,
    ):
        super().__init__()
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer(lowercase_tokens=True),
        }
        self._mention_indexers = mention_indexers or {
            "mentions": SingleIdTokenIndexer(),
        }
        self._mention_tokenizer = mention_tokenizer or WordTokenizer(
            word_splitter=JustSpacesWordSplitter(),
        )
        self._fact_lookup: Optional[Dict[int, Fact]] = None

    @overrides
    def _read(self, file_path: str):
        """
        file_path should point to a curiosity dialog file. In addition,
        the directory that contains that file should also contain the
        sqlite database associated with the dialogs named as below
        - wiki_sql.sqlite.db

        The intent is that there are
        """
        with open(file_path) as f:
            dataset = json.load(f)
            dialogs = dataset["dialogs"]

        directory = os.path.dirname(file_path)
        db_path = os.path.join(directory, "wiki_sql.sqlite.db")
        engine, session = create_sql(db_path)
        facts = session.query(Fact).all()
        self._fact_lookup = {f.id: f for f in facts}
        verify_checksum(dataset["db_checksum"], db_path)
        # store = CuriosityStore(db_path)
        # fact_lookup = store.get_fact_lookup()
        # TODO: Add in facts
        for _, d in enumerate(dialogs):
            yield self.text_to_instance(d)
        session.close()

    @overrides
    def text_to_instance(self, dialog: Dict, ignore_fact: bool = False):
        msg_texts = []
        msg_senders = []
        msg_likes = []
        msg_acts = []
        msg_act_mask = []
        msg_facts = []
        msg_fact_labels = []
        metadata_fact_labels = []
        if len(dialog["messages"]) == 0:
            raise ValueError("There are no dialog messages")

        known_entities = [
            Token(text="ENTITY/" + t.replace(" ", "_"), idx=idx)
            for idx, t in enumerate(dialog["known_entities"])
        ]
        if len(known_entities) == 0:
            known_entities.append(Token(text="@@YOUKNOWNOTHING@@", idx=0))
        known_entities_field = TextField(known_entities, self._mention_indexers)

        focus_entity = dialog["focus_entity"]
        focus_entity_field = TextField(
            [Token(text="ENTITY/" + focus_entity.replace(" ", "_"), idx=0)],
            self._mention_indexers,
        )
        for msg in dialog["messages"]:
            tokenized_msg = self._tokenizer.tokenize(msg["message"])
            msg_texts.append(TextField(tokenized_msg, self._token_indexers))
            msg_senders.append(0 if msg["sender"] == USER else 1)
            msg_likes.append(
                LabelField(
                    "liked" if msg["liked"] else "not_liked",
                    label_namespace="like_labels",
                )
            )
            if msg["dialog_acts"] is None:
                dialog_acts = ["@@NODA@@"]
                act_mask = 0
            else:
                dialog_acts = msg["dialog_acts"]
                act_mask = 1
            dialog_acts_field = MultiLabelFieldListCompat(
                dialog_acts, label_namespace=DIALOG_ACT_LABELS
            )
            msg_acts.append(dialog_acts_field)
            msg_act_mask.append(act_mask)
            curr_facts_text = []
            curr_facts_labels = []
            curr_metadata_fact_labels = []
            if msg["sender"] == ASSISTANT:
                for idx, f in enumerate(msg["facts"]):
                    if ignore_fact:
                        fact_text = "dummy fact"
                    else:
                        fact = self._fact_lookup[f["fid"]]
                        fact_text = fact.text
                    # TODO: These are already space tokenized
                    tokenized_fact = self._tokenizer.tokenize(fact_text)
                    # 99% of text length is 77
                    tokenized_fact = tokenized_fact[:80]
                    curr_facts_text.append(
                        TextField(tokenized_fact, self._token_indexers)
                    )
                    if f["used"]:
                        curr_facts_labels.append(idx)
                        curr_metadata_fact_labels.append(idx)
            else:
                # Users don't have facts, but lets avoid divide by zero
                curr_facts_text.append(
                    TextField([Token(text="@@NOFACT@@", idx=0)], self._token_indexers)
                )

            msg_facts.append(ListField(curr_facts_text))
            # Add in a label if there are no correct indices
            if len(curr_facts_labels) == 0:
                curr_metadata_fact_labels.append(-1)
            n_facts = len(curr_facts_text)
            fact_label_arr = np.zeros(n_facts, dtype=np.float32)
            if len(curr_facts_labels) > 0:
                fact_label_arr[curr_facts_labels] = 1
            msg_fact_labels.append(ArrayField(fact_label_arr, dtype=np.float32))
            metadata_fact_labels.append(curr_metadata_fact_labels)

        return Instance(
            {
                "messages": ListField(msg_texts),
                "facts": ListField(msg_facts),
                "fact_labels": ListField(msg_fact_labels),
                "likes": ListField(msg_likes),
                "dialog_acts": ListField(msg_acts),
                "dialog_acts_mask": to_long_field(msg_act_mask),
                "senders": to_long_field(msg_senders),
                "focus_entity": focus_entity_field,
                "known_entities": known_entities_field,
                "metadata": MetadataField(
                    {
                        "dialog_id": dialog["dialog_id"],
                        "n_message": len(msg_texts),
                        "fact_labels": metadata_fact_labels,
                        "known_entities": dialog["known_entities"],
                        "focus_entity": dialog["focus_entity"],
                    }
                ),
            }
        )


@DatasetReader.register("curiosity_paraphrase")
class CuriosityParaphraseReader(DatasetReader):
    def __init__(
        self,
        filter_user_messages: bool = True,
        filter_empty_facts: bool = True,
        lazy: bool = False,
    ) -> None:

        super().__init__(lazy)
        self._tokenizer = WordTokenizer(
            word_splitter=JustSpacesWordSplitter(),
        )
        self._token_indexers = {"tokens": SingleIdTokenIndexer()}
        self._fact_lookup: Optional[Dict[int, Fact]] = None
        self._filter_user_messages = filter_user_messages
        self._filter_empty_facts = filter_empty_facts

    @overrides
    def _read(self, file_path: str):
        """
        file_path should point to a curiosity dialog file. In addition,
        the directory that contains that file should also contain the
        sqlite database associated with the dialogs named as below
        - wiki_sql.sqlite.db
        """
        with open(file_path) as f:
            dataset = json.load(f)
            dialogs = dataset["dialogs"]

        directory = os.path.dirname(file_path)
        db_path = os.path.join(directory, "wiki_sql.sqlite.db")
        engine, session = create_sql(db_path)
        facts = session.query(Fact).all()
        self._fact_lookup = {f.id: f for f in facts}
        verify_checksum(dataset["db_checksum"], db_path)
        # store = CuriosityStore(db_path)
        # fact_lookup = store.get_fact_lookup()
        # TODO: Add in facts
        for d in dialogs:
            for msg in d["messages"]:
                # Filter out user messages
                if self._filter_user_messages and msg["sender"] == USER:
                    continue

                # Filter out messages without paired facts, if required
                if self._filter_empty_facts:
                    facts_used = [f for f in msg["facts"] if f["used"]]
                    if len(facts_used) == 0:
                        continue

                yield self.text_to_instance(msg, d)

        session.close()

    @overrides
    def text_to_instance(self, msg: Dict, d: Dict):
        # (1) Prepare facts
        # Set max length of each fact text: 300 characters
        fact_texts = [
            self._fact_lookup[f["fid"]].text[:300] for f in msg["facts"] if f["used"]
        ]

        # Aggregate facts
        aggregated_fact_text = " ".join(fact_texts)

        # If it doesn't have any fact, put a default symbol
        if aggregated_fact_text == "":
            aggregated_fact_text = "@@NOFACT@@"

        # Wrap each sentence with start and end symbols
        aggregated_fact_text = "{start_symbol} {text} {end_symbol}".format(
            start_symbol=START_SYMBOL, text=aggregated_fact_text, end_symbol=END_SYMBOL
        )

        # Tokenize facts
        tokenized_fact = self._tokenizer.tokenize(aggregated_fact_text)[:150]

        # (2) Prepare messages
        message = msg["message"] if msg["message"] != "" else "@@NOMESSAGE@@"

        # Wrap each sentence with start and end symbols
        message = "{start_symbol} {text} {end_symbol}".format(
            start_symbol=START_SYMBOL, text=message, end_symbol=END_SYMBOL
        )

        # Tokenize
        tokenized_message = self._tokenizer.tokenize(message)[:150]

        # (3) Prepare dialog acts
        dialog_acts = (
            msg["dialog_acts"] if msg["dialog_acts"] is not None else ["@@NODA@@"]
        )

        # (4) Prepare sender information
        sender = "user" if msg["sender"] == USER else "teacher"

        return Instance(
            {
                "source_tokens": TextField(tokenized_fact, self._token_indexers),
                "target_tokens": TextField(tokenized_message, self._token_indexers),
                "dialog_acts": MultiLabelField(
                    dialog_acts, label_namespace=DIALOG_ACT_LABELS
                ),
                "sender": LabelField(sender, label_namespace="sender"),
                "metadata": MetadataField(
                    {
                        "dialog_id": d["dialog_id"],
                        "n_message": len(d["messages"]),
                    }
                ),
            }
        )


@DatasetReader.register("fact_paraphrase")
class FactParaphraseReader(DatasetReader):
    def __init__(
        self,
        filter_empty_facts: bool = True,
        lazy: bool = False,
    ) -> None:

        super().__init__(lazy)
        self._tokenizer = WordTokenizer(
            word_splitter=JustSpacesWordSplitter(),
        )
        self._token_indexers = {"tokens": SingleIdTokenIndexer()}
        self._fact_lookup: Optional[Dict[int, Fact]] = None
        self._filter_empty_facts = filter_empty_facts

    @overrides
    def _read(self, file_path: str):
        """
        file_path should point to a fact paraphrase dataset file.
        It assumes the following format: {fact}\t{paraphrased}\n ...
        """
        with open(file_path) as f:
            reader = csv.reader(f, delimiter="\t")

            for row in reader:
                if self._filter_empty_facts and row[0] == "":
                    continue
                yield self.text_to_instance(row)

    @overrides
    def text_to_instance(self, row: List):
        # (1) Prepare facts
        # Set max length of each fact text: 300 characters
        fact = row[0][:300]

        # If it doesn't have any fact, put a default symbol
        if fact == "":
            fact = "@@NOFACT@@"

        # Tokenize facts
        tokenized_fact = (
            [Token(START_SYMBOL)]
            + self._tokenizer.tokenize(fact)[:150]
            + [Token(END_SYMBOL)]
        )

        # (2) Prepare the paraphrased message
        message = row[1]

        # Tokenize
        tokenized_message = (
            [Token(START_SYMBOL)]
            + self._tokenizer.tokenize(message)[:150]
            + [Token(END_SYMBOL)]
        )

        # (3) Prepare dialog acts
        dialog_acts = ["@@NODA@@"]

        # (4) Prepare sender information
        sender = "teacher"

        return Instance(
            {
                "source_tokens": TextField(tokenized_fact, self._token_indexers),
                "target_tokens": TextField(tokenized_message, self._token_indexers),
                "dialog_acts": MultiLabelField(
                    dialog_acts, label_namespace="dialog_acts"
                ),
                "sender": LabelField(sender, label_namespace="sender"),
                "metadata": MetadataField(
                    {
                        "dialog_id": -1,
                        "n_message": -1,
                    }
                ),
            }
        )
