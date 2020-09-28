#!/usr/bin/env bash

# Similarly to data.sh, this script is mainly documentation of
# how to run all the experiments

conda activate curiosity

# This computes the like baseline and saves scores to
# like_majority_<fold>_metrics.json
./cli majority experiments/

# da_majority_<fold>_metrics.json
./cli majority-da experiments/

# policy_majority_<fold>_metrics.json
./cli majority-policy experiments/

# Compute tfidf baseline
./cli fact-tfidf dialog_data/tfidf.pickle  dialog_data/wiki_sql.sqlite.db experiments/

# Generate allennlp training configs
# These appear in `configs/generated`
./cli gen-configs experiments/
