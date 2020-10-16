#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.

import os
import json
from itertools import cycle
import click
import _jsonnet

from curiosity.stats import (
    MajorityLikes,
    TfidfFactBaseline,
    MajorityDialogActs,
    MajorityPolicyActs,
    save_metrics,
    fact_length_stats,
)
from curiosity.util import get_logger


log = get_logger(__name__)


TRAIN_DIALOGS = "dialog_data/curiosity_dialogs.train.json"
VAL_DIALOGS = "dialog_data/curiosity_dialogs.val.json"
TEST_DIALOGS = "dialog_data/curiosity_dialogs.test.json"
ZERO_DIALOGS = "dialog_data/curiosity_dialogs.test_zero.json"


@click.group()
def cli():
    pass


@cli.command()
@click.argument("metrics_dir")
def majority(metrics_dir):
    """
    Obtain a majority baseline for like prediction
    """
    model = MajorityLikes()
    model.train(TRAIN_DIALOGS)
    val_score = model.score(VAL_DIALOGS)
    test_score = model.score(TEST_DIALOGS)
    zero_score = model.score(ZERO_DIALOGS)
    log.info("Like prediction")
    log.info(f"Validation Score: {val_score}")
    log.info(f"Test Score: {test_score}")
    log.info(f"Zero Score: {zero_score}")
    save_metrics(
        {
            "best_validation_like_accuracy": val_score,
        },
        os.path.join(metrics_dir, "like_majority_val_metrics.json"),
    )
    save_metrics(
        {
            "best_validation_like_accuracy": test_score,
        },
        os.path.join(metrics_dir, "like_majority_test_metrics.json"),
    )
    save_metrics(
        {
            "best_validation_like_accuracy": zero_score,
        },
        os.path.join(metrics_dir, "like_majority_zero_metrics.json"),
    )


@cli.command()
@click.argument("metrics_dir")
def majority_da(metrics_dir):
    """
    Obtain a majority baseline for dialog acts prediction
    """
    model = MajorityDialogActs()
    model.train(TRAIN_DIALOGS)
    val_score = model.score(VAL_DIALOGS)
    test_score = model.score(TEST_DIALOGS)
    zero_score = model.score(ZERO_DIALOGS)
    log.info("Dialog Acts prediction")
    log.info(f"Validation Score: {val_score}")
    log.info(f"Test Score: {test_score}")
    log.info(f"Zero Score: {zero_score}")
    save_metrics(
        {
            "best_validation_da_micro_f1": val_score,
        },
        os.path.join(metrics_dir, "da_majority_val_metrics.json"),
    )
    save_metrics(
        {
            "best_validation_da_micro_f1": test_score,
        },
        os.path.join(metrics_dir, "da_majority_test_metrics.json"),
    )
    save_metrics(
        {
            "best_validation_da_micro_f1": zero_score,
        },
        os.path.join(metrics_dir, "da_majority_zero_metrics.json"),
    )


@cli.command()
@click.argument("metrics_dir")
def majority_policy(metrics_dir):
    """
    Obtain a majority baseline for policy acts prediction
    """
    model = MajorityPolicyActs()
    model.train(TRAIN_DIALOGS)
    val_score = model.score(VAL_DIALOGS)
    test_score = model.score(TEST_DIALOGS)
    zero_score = model.score(ZERO_DIALOGS)
    log.info("Policy Acts prediction")
    log.info(f"Validation Score: {val_score}")
    log.info(f"Test Score: {test_score}")
    log.info(f"Zero Score: {zero_score}")
    save_metrics(
        {
            "best_validation_policy_micro_f1": val_score,
        },
        os.path.join(metrics_dir, "policy_majority_val_metrics.json"),
    )
    save_metrics(
        {
            "best_validation_policy_micro_f1": test_score,
        },
        os.path.join(metrics_dir, "policy_majority_test_metrics.json"),
    )
    save_metrics(
        {
            "best_validation_policy_micro_f1": zero_score,
        },
        os.path.join(metrics_dir, "policy_majority_zero_metrics.json"),
    )


@cli.command()
@click.argument("tfidf_path")
@click.argument("wiki_sql_path")
@click.argument("metrics_dir")
def fact_tfidf(tfidf_path, wiki_sql_path, metrics_dir):
    """
    Train and evaluate a tfidf baseline in the same format as the allennlp
    models
    """
    model = TfidfFactBaseline(tfidf_path, wiki_sql_path=wiki_sql_path)
    val_score = model.score(VAL_DIALOGS)
    test_score = model.score(TEST_DIALOGS)
    zero_score = model.score(ZERO_DIALOGS)
    log.info("Fact Prediction")
    log.info(f"Validation Score: {val_score}")
    log.info(f"Test Score: {test_score}")
    log.info(f"Zero Score: {zero_score}")
    save_metrics(
        {"best_validation_fact_mrr": val_score},
        os.path.join(metrics_dir, "mrr_tfidf_val_metrics.json"),
    )
    save_metrics(
        {"best_validation_fact_mrr": test_score},
        os.path.join(metrics_dir, "mrr_tfidf_test_metrics.json"),
    )
    save_metrics(
        {"best_validation_fact_mrr": zero_score},
        os.path.join(metrics_dir, "mrr_tfidf_zero_metrics.json"),
    )


@cli.command()
@click.argument("data_path")
def fact_lengths(data_path):
    fact_length_stats(data_path)


@cli.command()
@click.option("--gpu", multiple=True)
@click.argument("metrics_dir")
def gen_configs(gpu, metrics_dir):
    """
    Create the configuration files for the different models.
    This is separate from hyper parameter tuning and directly
    corresponds to models in the paper table

    The gpu flag can be taken multiple times and indicates to write
    jobs configured to use
    those gpus.
    """
    gpu_list = []
    if isinstance(gpu, (tuple, list)):
        if len(gpu) == 0:
            gpu_list.append(-1)
        else:
            for e in gpu:
                gpu_list.append(int(e))
    elif isinstance(gpu, int):
        gpu_list.append(gpu)
    elif isinstance(gpu, str):
        gpu_list.append(int(gpu))
    else:
        raise ValueError("wrong input type")
    gpu_list = cycle(gpu_list)
    # Note: key should be valid in a unix filename
    # Values must be strings that represent jsonnet "code"
    # These names are shared to the figure plotting code since the filename
    # is based on it
    all_configs = {
        # This is the full model, so default params
        "glove_bilstm": {},
        # Ablations leaving on out
        "glove_bilstm-known": {"disable_known_entities": "true"},
        # Completely ablate out anything related to dialog acts
        "glove_bilstm-da": {"disable_dialog_acts": "true"},
        # Completely ablate out anything related to likes
        "glove_bilstm-like": {"disable_likes": "true"},
        # Completley ablate out anything related to facts
        "glove_bilstm-facts": {"disable_facts": "true"},
        # This is the full model, so default params
        "bert": {"use_glove": "false", "use_bert": "true"},
        # Ablations leaving on out
        "bert-known": {
            "disable_known_entities": "true",
            "use_glove": "false",
            "use_bert": "true",
        },
        # Completely ablate out anything related to dialog acts
        "bert-da": {
            "disable_dialog_acts": "true",
            "use_glove": "false",
            "use_bert": "true",
        },
        # Completely ablate out anything related to likes
        "bert-like": {
            "disable_likes": "true",
            "use_glove": "false",
            "use_bert": "true",
        },
        # Completley ablate out anything related to facts
        "bert-facts": {
            "disable_facts": "true",
            "use_glove": "false",
            "use_bert": "true",
        },
    }
    with open("run_allennlp.sh", "w") as exp_f:
        exp_f.write("#!/usr/bin/env bash\n")
        for name, conf in all_configs.items():
            model_conf: str = _jsonnet.evaluate_file(
                "configs/model.jsonnet", tla_codes=conf
            )
            config_path = os.path.join("configs/generated", name + ".json")
            model_path = os.path.join("models", name)
            with open(config_path, "w") as f:
                f.write(model_conf)

            job_gpu = next(gpu_list)
            if job_gpu != -1:
                gpu_option = (
                    " -o '" + json.dumps({"trainer": {"cuda_device": job_gpu}}) + "'"
                )
            else:
                gpu_option = ""
            exp_f.write(f"# Experiments for: {name}\n")
            exp_f.write(
                f"allennlp train --include-package curiosity -s {model_path} -f {config_path}{gpu_option}\n"
            )

            val_out = os.path.join(metrics_dir, f"{name}_val_metrics.json")
            test_out = os.path.join(metrics_dir, f"{name}_test_metrics.json")
            zero_out = os.path.join(metrics_dir, f"{name}_zero_metrics.json")
            exp_f.write(
                f"allennlp evaluate --include-package curiosity{gpu_option} --output-file {val_out} {model_path} {VAL_DIALOGS}\n"
            )
            exp_f.write(
                f"allennlp evaluate --include-package curiosity{gpu_option} --output-file {test_out} {model_path} {TEST_DIALOGS}\n"
            )
            exp_f.write(
                f"allennlp evaluate --include-package curiosity{gpu_option} --output-file {zero_out} {model_path} {ZERO_DIALOGS}\n"
            )
            exp_f.write("\n")


@cli.command()
@click.argument("emb_path")
@click.argument("out_path")
def filter_emb(emb_path, out_path):
    """
    Given a path to a valid pretrained embeddings file from wikipedia2vec,
    filter out anything that is not an entity: IE not prefixed with ENTITY/
    """
    with open(emb_path) as in_f:
        rows = []
        _, dim = next(in_f).strip().split()
        dim = dim
        for line in in_f:
            if line.startswith("ENTITY/"):
                rows.append(line)
    with open(out_path, "w") as out_f:
        out_f.write(f"{len(rows)} {dim}\n")
        for r in rows:
            out_f.write(r)


if __name__ == "__main__":
    cli()
