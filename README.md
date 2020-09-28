# Information Seeking in the Spirit of Learning: a Dataset for Conversational Curiosity

This repository contains code for our EMNLP 2020 paper which you can cite like this:

```
@inproceedings{rodriguez2020curiosity,
    title = {Information Seeking in the Spirit of Learning: a Dataset for Conversational Curiosity},
    author = {Pedro Rodriguez and Paul Crook and Seungwhan Moon and Zhiguang Wang},
    year = 2020,
    booktitle = {Empirical Methods in Natural Language Processing}
}
```

## Structure

The code for our paper is split into two parts: (1) model code used to run experiments and (2) code/latex that generates the publication PDF file.
The published PDF was created by running the experiments and then exporting the experimental results to the paper code, which is compiled to the paper itself.

## Model Code

Our model code is written using `pytorch` and `allennlp`.
To run it, you'll need to install some software and download some data.


### Installation

1. Install a recent version of anaconda python https://www.anaconda.com/distribution/

For CPU:

1. Create an anconda environment `conda env create -f environment.yaml` (creates an environment named curiosity)
2. Activate the environment, in fish shell this is `conda activate curiosity`

For GPU (not changed environemnt file):

1. Create an anconda environment `conda env create -f environment_gpu.yaml` (creates an environment named curiosity)
2. Activate the environment, in fish shell this is `conda activate curiosity`

### Data

Dialog data is stored in `dialog_data`, so you should not need to download anything more.
Our data is also available at TODO.

#### File Description

* `curiosity_dialogs.{train, val, test, test_zero}.json`: Dialogs corresponding to each data fold
* `wiki_sql.sqlite.db`: A sqlite database storing our processsed version of the Wikipedia subset that we use
* `fact_db_links.json`: A json file containing an entity linked version of our Wikipedia data


#### Entity Linked Facts

Although the `wiki_sql.sqlite.db` stores the facts, it does not store the positions of each entity link.
For this, you need to use the jsonlines file from the entity linker.

#### Wikipedia2Vec

The file `wiki2vec_entity_100d.txt` is the output of running the following steps:

1. Download the embeddings with `wget http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_100d.txt.bz2`
2. Decompress: `bzip2 -d enwiki_20180420_100d.txt.bz2`
3. Filter out non-entities with:
4. `./cli filter-emb enwiki_20180420_100d.txt wiki2vec_entity_100d.txt`

## Training and Evaluating Models

Models are run using a combination of the `allennlp train`, `allennlp evaluate`, and `./cli` command (in this repository).

In our paper, we vary models according to two axes:
- Text encoder: glove+lstm or bert
- Feature ablations: everything and leave one out

The configuration `configs/model.jsonnet` is the parent configuration. This can be converted into the set of configurations in the paper by running:

```bash
$ ./cli gen-configs /data/users/par/experiments/
```

This generates configurations in `configs/generated/` and a run file `run_allennlp.sh` that lists the correct command to run each model variant. Generally, the commands look like this:

```bash
$ allennlp train --include-package curiosity -s /data/users/par/models/glove_bilstm -f configs/generated/glove_bilstm.json
$ allennlp evaluate --include-package curiosity --output-file /data/users/par/experiments/glove_bilstm_val_metrics.json /data/users/par/models/glove_bilstm /data/users/par/dialog-data/folded/1022/curiosity_dialogs.val.json
$ allennlp evaluate --include-package curiosity --output-file /data/users/par/experiments/glove_bilstm_test_metrics.json /data/users/par/models/glove_bilstm /data/users/par/dialog-data/folded/1022/curiosity_dialogs.test.json
$ allennlp evaluate --include-package curiosity --output-file /data/users/par/experiments/glove_bilstm_zero_metrics.json /data/users/par/models/glove_bilstm /data/users/par/dialog-data/folded/1022/curiosity_dialogs.test_zero.json
```

### Running on GPU
In addition to installing the gpu variant of the environment, to use the GPU you need to pass: `-o '{"trainer": {"cuda_device": 0}}'` to the `allennlp train` command. You should change the device number to match an open GPU.

## Export to Paper
The configuration generator also properly names files so that if you copy files with `ssh` as shown below, the results will automagically update the next time you run `make 2020_acl_curiosity.paper.pdf` in the repo:



```bash
scp '/data/users/par/experiments/*' ~/code/curiosity-paper/2020_acl_curiosity/data/experiments/
```
Then be sure to commit those updates to the repository.

TODO: Update this section to explain thats how to reproduce paper results

## Tests

Run `pytest` to run unit tests for the loss, metrics, and reader.

## Adding new module

The easiest way to modify the model is to:
- If possible, add a parameter to `curiosity.model.CuriosityModel` that is a general version of it (eg, `Seq2VecEncoder` or `FactRanker`)
- Implement a module that implements its api (or use one/combination in allennlp), EG `curiosity.nn.MeanLogitRanker` implements `curiosity.nn.FactRanker`
- In the config `configs/bilstm.jsonnet`, name the implementaiton to use, eg the `fact_ranker` entry

## FAQ

1. Is the data collection interface open source? No, unfortunately that is tied to internal systems so difficult to open source.