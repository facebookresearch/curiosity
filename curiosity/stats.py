#!/usr/bin/env python3
"""
This file computes baseline accuracies based on majority
based voting
"""
from typing import Dict, Optional, List
import json
import numpy as np
import pandas as pd
from allennlp.data.tokenizers.token import Token
from curiosity.reader import CuriosityDialogReader
from curiosity.similarity import Similarity
from curiosity.util import get_logger


log = get_logger(__name__)

ASSISTANT_IDX = 1


def save_metrics(metrics: Dict, out_path: str):
    """
    Save an allennlp compatible metrics dictionary
    """
    out_dict = {
        "best_epoch": 0,
        "peak_cpu_memory_MB": 0,
        "training_duration": "0:00:0",
        "training_start_epoch": 0,
        "training_epochs": 0,
        "epoch": 0,
        "training_like_accuracy": 0.0,
        "training_loss": 0.0,
        "training_cpu_memory_MB": 0.0,
        "validation_like_accuracy": 0.0,
        "validation_loss": 0.0,
        "best_validation_like_accuracy": 0.0,
        "best_validation_loss": 0.0,
    }
    for key, val in metrics.items():
        out_dict[key] = val

    with open(out_path, "w") as f:
        json.dump(out_dict, f)


class MajorityLikes:
    def __init__(self):
        self._n_total_assistant_msgs = 0
        self._n_liked_assistant_msgs = 0
        self._like_all = True

    def train(self, data_path: str) -> None:
        log.info(f"Training majority classifier with: {data_path}")
        self._n_total_assistant_msgs = 0
        self._n_liked_assistant_msgs = 0
        n_messages = 0
        dialogs = CuriosityDialogReader().read(data_path)
        log.info(f"N Dialogs: {len(dialogs)}")
        for d in dialogs:
            dialog_senders = d["senders"].array
            dialog_likes = d["likes"]
            for sender, liked in zip(dialog_senders, dialog_likes):
                # Only care about assistant messages
                if sender == ASSISTANT_IDX:
                    if liked.label == "liked":
                        self._n_liked_assistant_msgs += 1
                    self._n_total_assistant_msgs += 1
                n_messages += 1
        self._n_total_assistant_msgs = max(1, self._n_total_assistant_msgs)
        log.info(f"N Liked Assistant Messages: {self._n_liked_assistant_msgs}")
        log.info(f"N Total Assistant Messages: {self._n_total_assistant_msgs}")
        log.info(f"N Total Messages: {n_messages}")
        if (self._n_liked_assistant_msgs / self._n_total_assistant_msgs) > 0.5:
            self._like_all = True
        else:
            self._like_all = False
        log.info(f"Majority Class Liked: {self._like_all}")

    def score(self, data_path: str) -> float:
        log.info(f"Scoring majority classifier with: {data_path}")
        dialogs = CuriosityDialogReader().read(data_path)
        log.info(f"N Dialogs: {len(dialogs)}")
        correct = 0
        total = 0
        n_messages = 0
        for d in dialogs:
            dialog_senders = d["senders"].array
            dialog_likes = d["likes"]
            for sender, liked in zip(dialog_senders, dialog_likes):
                if sender == ASSISTANT_IDX:
                    label = liked.label
                    # If liked and majority class in training was liked
                    if label == "liked" and self._like_all:
                        correct += 1
                    # If not liked and majority class in training was not liked
                    elif label == "liked" and not self._like_all:
                        correct += 1
                    total += 1
                n_messages += 1

        log.info(f"N Correct Assistant Messages: {correct}")
        log.info(f"N Total Assistant Messages: {total}")
        log.info(f"N Total Messages: {n_messages}")
        total = max(1, total)
        return correct / total


class MajorityDialogActs:
    def __init__(self):
        self._n_total_assistant_msgs = 0
        self._n_total_acts = 0

        self._count_per_turn = {}
        self._majority_per_turn = {}

        self._count = {}
        self._majority = None

    def train(self, data_path: str) -> None:
        log.info(f"Training majority classifier with: {data_path}")
        self._n_total_assistant_msgs = 0
        n_messages = 0
        dialogs = CuriosityDialogReader().read(data_path)
        log.info(f"N Dialogs: {len(dialogs)}")
        for d in dialogs:
            dialog_senders = d["senders"].array
            dialog_acts_list = d["dialog_acts"]

            for i in range(len(dialog_senders)):
                sender = dialog_senders[i]
                acts = dialog_acts_list[i].labels

                # Only care about assistant messages
                if sender != ASSISTANT_IDX:
                    # Histogram stat per turn
                    if i not in self._count_per_turn:
                        self._count_per_turn[i] = {}

                    for act in acts:
                        # Histogram stat per turn
                        self._count_per_turn[i][act] = (
                            self._count_per_turn[i].get(act, 0) + 1
                        )

                        # Histogram stat overall
                        self._count[act] = self._count.get(act, 0) + 1

                        # Total count
                        self._n_total_acts += 1

                    self._n_total_assistant_msgs += 1
                n_messages += 1
        self._n_total_assistant_msgs = max(1, self._n_total_assistant_msgs)
        log.info(f"N Total User Messages: {self._n_total_acts}")
        log.info(f"N Total Acts: {self._n_total_assistant_msgs}")
        log.info(f"N Total Messages: {n_messages}")

        # Sort count overall
        lst = [(count, act) for act, count in self._count.items()]
        lst.sort(reverse=True)

        # Majority act in this turn
        self._majority = lst[0][1]

        for turn_idx, act_stat in self._count_per_turn.items():
            # Sort count_per_turn for each turn_idx
            lst = [(count, act) for act, count in act_stat.items()]
            lst.sort(reverse=True)

            if len(lst) != 0:
                majority_act = lst[0][1]
            else:
                majority_act = self._majority

            # Majority act in this turn
            self._majority_per_turn[turn_idx] = majority_act
            print("Turn: %d, Majority Act: %s" % (turn_idx, majority_act))

        log.info(f"Majority Act: {self._majority}")
        log.info(f"Majority Map: {self._majority_per_turn}")
        log.info(f"Count Map Per Turn: {self._count_per_turn}")
        log.info(f"Count Map: {self._count}")

    def score(self, data_path: str) -> float:
        log.info(f"Scoring majority classifier with: {data_path}")
        dialogs = CuriosityDialogReader().read(data_path)
        log.info(f"N Dialogs: {len(dialogs)}")
        correct = 0
        total = 0
        n_messages = 0
        for d in dialogs:
            dialog_senders = d["senders"].array
            dialog_acts_list = d["dialog_acts"]

            for i in range(len(dialog_senders)):
                sender = dialog_senders[i]
                acts = dialog_acts_list[i].labels

                if sender != ASSISTANT_IDX:
                    for act in acts:
                        if i in self._majority_per_turn:
                            if act == self._majority_per_turn[i]:
                                correct += 1
                        else:
                            if act == self._majority:
                                correct += 1

                    total += len(acts)
                    n_messages += 1

        log.info(f"N Correct Acts: {correct}")
        log.info(f"N Total Acts: {total}")
        log.info(f"N Total Messages: {n_messages}")
        total = max(1, total)
        n_messages = max(1, n_messages)
        p = correct / n_messages  # assumes 1 prediction per message
        r = correct / total
        f1 = 2 * (p * r) / (p + r)
        return f1


class MajorityPolicyActs:
    def __init__(self):
        self._n_total_assistant_msgs = 0
        self._n_total_acts = 0

        self._count_per_turn = {}
        self._majority_per_turn = {}

        self._count = {}
        self._majority = None

    def train(self, data_path: str) -> None:
        log.info(f"Training majority classifier with: {data_path}")
        self._n_total_assistant_msgs = 0
        n_messages = 0
        dialogs = CuriosityDialogReader().read(data_path)
        log.info(f"N Dialogs: {len(dialogs)}")
        for d in dialogs:
            dialog_senders = d["senders"].array
            dialog_acts_list = d["dialog_acts"]

            for i in range(len(dialog_senders)):
                sender = dialog_senders[i]
                acts = dialog_acts_list[i].labels

                # Histogram stat per turn
                if i not in self._count_per_turn:
                    self._count_per_turn[i] = {}

                # Only care about assistant messages
                if sender == ASSISTANT_IDX:
                    for act in acts:
                        # Histogram stat per turn
                        self._count_per_turn[i][act] = (
                            self._count_per_turn[i].get(act, 0) + 1
                        )

                        # Histogram stat overall
                        self._count[act] = self._count.get(act, 0) + 1

                        # Total count
                        self._n_total_acts += 1

                    self._n_total_assistant_msgs += 1
                n_messages += 1
        self._n_total_assistant_msgs = max(1, self._n_total_assistant_msgs)
        log.info(f"N Total Assistant Messages: {self._n_total_acts}")
        log.info(f"N Total Acts: {self._n_total_assistant_msgs}")
        log.info(f"N Total Messages: {n_messages}")

        # Sort count overall
        lst = [(count, act) for act, count in self._count.items()]
        lst.sort(reverse=True)

        # Majority act in this turn
        self._majority = lst[0][1]

        for turn_idx, act_stat in self._count_per_turn.items():
            # Sort count_per_turn for each turn_idx
            lst = [(count, act) for act, count in act_stat.items()]
            lst.sort(reverse=True)

            if len(lst) != 0:
                majority_act = lst[0][1]
            else:
                majority_act = self._majority

            # Majority act in this turn
            self._majority_per_turn[turn_idx] = majority_act
            print("Turn: %d, Majority Act: %s" % (turn_idx, majority_act))

        log.info(f"Majority Act: {self._majority}")
        log.info(f"Majority Map: {self._majority_per_turn}")
        log.info(f"Count Map Per Turn: {self._count_per_turn}")
        log.info(f"Count Map: {self._count}")

    def score(self, data_path: str) -> float:
        log.info(f"Scoring majority classifier with: {data_path}")
        dialogs = CuriosityDialogReader().read(data_path)
        log.info(f"N Dialogs: {len(dialogs)}")
        correct = 0
        total = 0
        n_messages = 0
        for d in dialogs:
            dialog_senders = d["senders"].array
            dialog_acts_list = d["dialog_acts"]

            for i in range(len(dialog_senders)):
                sender = dialog_senders[i]
                acts = dialog_acts_list[i].labels

                if sender == ASSISTANT_IDX:
                    for act in acts:
                        if i in self._majority_per_turn:
                            if act == self._majority_per_turn[i]:
                                correct += 1
                        else:
                            if act == self._majority:
                                correct += 1

                    total += len(acts)
                n_messages += 1

        log.info(f"N Correct Acts: {correct}")
        log.info(f"N Total Acts: {total}")
        log.info(f"N Total Messages: {n_messages}")
        total = max(1, total)
        n_messages = max(1, n_messages)
        p = correct / n_messages  # assumes 1 prediction per message
        r = correct / total
        f1 = 2 * (p * r) / (p + r)
        return f1


def tokens_to_str(tokens: List[Token]) -> str:
    return " ".join(t.text for t in tokens)


class TfidfFactBaseline:
    """
    Implements a simplistic baseline. This uses the tfidf vectorizer
    fit for ranking facts shown to annotators. The highest similarity
    fact is selected as the used fact. For metrics, precision and recall
    are both computed. In the data, more than one fact is rarely used,
    even if its possible to do.
    """

    def __init__(self, tfidf_path: str, wiki_sql_path: Optional[str] = None):
        self._similarity = Similarity()
        self._similarity.load(tfidf_path)

    def score(self, data_path: str):
        dialogs = CuriosityDialogReader().read(data_path)
        n_assistant_messages = 0
        all_rr = []
        for d in dialogs:
            msg_history = []
            dialog_senders = d["senders"].array
            dialog_facts = d["facts"]
            dialog_fact_labels = d["fact_labels"]
            dialog_messages = d["messages"]
            for msg, sender, facts, fact_labels in zip(
                dialog_messages, dialog_senders, dialog_facts, dialog_fact_labels
            ):
                if sender == ASSISTANT_IDX:
                    context = " ".join(msg_history)
                    fact_texts = [tokens_to_str(tokens) for tokens in facts]
                    doc_scores = self._similarity.score(context, fact_texts)
                    # First get a list where first position is maximal score
                    sorted_scores = np.argsort(-np.array(doc_scores))
                    exists_rel_doc = False
                    best_rank = None
                    for rel_idx in fact_labels.array:
                        if rel_idx != -1:
                            # Then find the rank + 1 of the relevant doc
                            exists_rel_doc = True
                            # import ipdb;ipdb.set_trace();
                            rank = np.where(sorted_scores == rel_idx)[0][0] + 1
                            # We only care about the best rank, if there are multiple
                            # relevant docs
                            if best_rank is None or rank < best_rank:
                                best_rank = rank

                    # Ignore this example if there is no relevant doc
                    if exists_rel_doc:
                        all_rr.append(1 / best_rank)
                    n_assistant_messages += 1

                # Only add the actually used message after prediction
                # Add user and assistant messages
                msg_text = tokens_to_str(msg.tokens)
                msg_history.append(msg_text)
        mean_rr = np.mean(all_rr)
        log.info(f"Msgs with Facts: {len(all_rr)}")
        log.info(f"Total Assistant Msgs: {n_assistant_messages}")
        log.info(f"MRR: {mean_rr}")
        return mean_rr


def fact_length_stats(data_path: str):
    dialogs = CuriosityDialogReader().read(data_path)
    fact_lengths = []
    for d in dialogs:
        for facts in d["facts"]:
            for f in facts:
                fact_lengths.append({"n_tokens": f.sequence_length()})
    df = pd.DataFrame(fact_lengths)
    summary = df.describe(percentiles=[0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99])
    log.info(f"Summary\n{summary}")
