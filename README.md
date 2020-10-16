# Information Seeking in the Spirit of Learning: a Dataset for Conversational Curiosity

This repository accompanies our EMNLP 2020 paper which you can cite like this:

```
@inproceedings{rodriguez2020curiosity,
    title = {Information Seeking in the Spirit of Learning: a Dataset for Conversational Curiosity},
    author = {Pedro Rodriguez and Paul Crook and Seungwhan Moon and Zhiguang Wang},
    year = 2020,
    booktitle = {Empirical Methods in Natural Language Processing}
}
```

To explore the dataset visit: [datasets.pedro.ai/curiosity](https://datasets.pedro.ai/curiosity)

For a summary of our work visit: [pedro.ai/curiosity](https://www.pedro.ai/curiosity)

## Structure

The project has three components:

1. The Curiosity dataset in `dialog_data/curiosity_dialogs.json`, with folded versions in `dialog_data/curiosity_dialogs.*.json`
2. Modeling code used in experiments
3. [Analysis, plotting, and latex code that generates the publication's PDF file](https://github.com/entilzha/publications).

### Data

#### File Description

* `curiosity_dialogs.{train, val, test, test_zero}.json`: Dialogs corresponding to each data fold
* `wiki_sql.sqlite.db`: A sqlite database storing our processed version of the Wikipedia subset that we use
* `fact_db_links.json`: A json file containing an entity linked version of our Wikipedia data. It stores the location of the entity link, which the database does not contain

#### Downloading Data
There are two ways to download our data.


First, you could clone our repository and use git lfs.

1. Install [git lfs](https://git-lfs.github.com)
2. Clone the repository: `https://github.com/facebookresearch/curiosity.git`
3. Run `git lfs pull`

Second, you can download from these URLs with tools like `wget`:

* https://obj.umiacs.umd.edu/curiosity/curiosity_dialogs.json
* https://obj.umiacs.umd.edu/curiosity/curiosity_dialogs.train.json
* https://obj.umiacs.umd.edu/curiosity/curiosity_dialogs.val.json
* https://obj.umiacs.umd.edu/curiosity/curiosity_dialogs.test.json
* https://obj.umiacs.umd.edu/curiosity/curiosity_dialogs.test_zero.json
* https://obj.umiacs.umd.edu/curiosity/fact_db_links.json
* https://obj.umiacs.umd.edu/curiosity/wiki2vec_entity_100d.txt
* https://obj.umiacs.umd.edu/curiosity/wiki_sql.sqlite.db


#### Re-creating Processed Data

We provide the inputs to our modeling experiments; to reproduce these inputs follow these instructions.

##### Wikipedia2Vec

The file `wiki2vec_entity_100d.txt` is the output of running the following steps:

1. Download the embeddings with `wget http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_100d.txt.bz2`
2. Decompress: `bzip2 -d enwiki_20180420_100d.txt.bz2`
3. Filter out non-entities with:
4. `./cli filter-emb enwiki_20180420_100d.txt wiki2vec_entity_100d.txt`


### Model Code

Our model code is written using `pytorch` and `allennlp`.
Before reproducing our experiments, you'll need to install some software.

#### Installation

Install a recent version of anaconda python https://www.anaconda.com/distribution/.
The canonical way to reproduce our experiments is with the [poetry configuration](https://python-poetry.org).
We also provide anaconda environment definitions, but the exact versions of all dependencies are not pinned so results may differ.

##### Poetry-based Instructions

1. Install [poetry](https://python-poetry.org)
2. Run `conda create -n curiosity python=3.7`
3. Run `conda activate curiosity` (fish shell)
4. Run `poetry install`
5. Before running any model commands, activate the environment with `poetry shell`

##### Anaconda-based Instructions

For CPU:

1. Create an anaconda environment `conda env create -f environment.yaml` (creates an environment named curiosity)
2. Activate the environment, in fish shell this is `conda activate curiosity`

For GPU:

1. Create an anaconda environment `conda env create -f environment_gpu.yaml` (creates an environment named curiosity)
2. Activate the environment, in fish shell this is `conda activate curiosity`

##### Docker-based Instructions

If you prefer using Docker for dependencies, we include a `Dockerfile` that builds all the required dependencies.
Note, that to enable GPU support you may need to use nvidia-docker and modify this file to install cuda dependencies

#### Training and Evaluating Models

Models are run using a combination of the `allennlp train`, `allennlp evaluate`, and `./cli` command (in this repository).

In our paper, we vary models according to two axes:
* Our `charm` model corresponds to `glove_bilstm`
* `glove_distributed` is the context-free version of `charm`
* The `bert` baseline corresponds to `e2e_bert`
* Names with like `glove_bilstm-feature` mean train `glove_bilstm` ablating (`-` minus) `feature`.

`allennlp` defines model configuration with `jsonnet` or `json` files.
In our work, we used the configuration files in `configs/generated/`:

* `glove_bilstm.json`
* `glove_distributed.json`
* `e2e_bert.json`

These configurations were generated from the parent configuration `configs/model.jsonnet`
To re-generate these, you can run this command:

```bash
$ ./cli gen-configs experiments/
```

This generates configurations in `configs/generated/` and a run file `run_allennlp.sh` that lists the correct command to run each model variant. Generally, the commands look like this:

```bash
$ allennlp train --include-package curiosity -s models/glove_bilstm -f configs/generated/glove_bilstm.json
$ allennlp evaluate --include-package curiosity --output-file experiments/glove_bilstm_val_metrics.json models/glove_bilstm dialog_data/curiosity_dialogs.val.json
$ allennlp evaluate --include-package curiosity --output-file experiments/glove_bilstm_test_metrics.json models/glove_bilstm dialog_data/curiosity_dialogs.test.json
$ allennlp evaluate --include-package curiosity --output-file experiments/glove_bilstm_zero_metrics.json models/glove_bilstm dialog_data/curiosity_dialogs.test_zero.json
```

By default, the configurations don't specify the cuda device so this must be passed in as an override like so:
* For `allennlp train`: `-o '{"trainer": {"cuda_device": 0}}'`
* For `allennlp evaluate`: `--cuda-device 0`


#### Export Experiments to Paper

The configuration generator also properly names files so that if you copy files with `ssh` as shown below, the results will automagically update the next time you run `make 2020_acl_curiosity.paper.pdf` in the repo:

By default, the scripts in `run_allennlp.sh` put models in `models/` and experimental results (metrics etc) in `experiments`.
Our code is designed so that copying the contents of `experiments` into a corresponding directory in the paper code will "import" the results into the paper.

```bash
# Local copy
cp 'experiments/*' ~/code/curiosity-paper/2020_emnlp_curiosity/data/experiments/
# remote copy
scp 'experiments/*' hostname:~/code/curiosity-paper/2020_emnlp_curiosity/data/experiments/
```

#### Running Tests

Run `pytest` to run unit tests for the loss, metrics, and reader.


## Paper Code

The code for the paper can be found here: github.com/entilzha/publications

## FAQ

1. Is the data collection interface open source? No, unfortunately that is tied to internal systems so it is difficult to open source. The interfaces were written in a combination of ReactJS and python/flask.
2. Who should I contact with questions? Please email Pedro Rodriguez at me@pedro.ai

## License

Curiosity is released under [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode), see [LICENSE](LICENSE) for details.
