#!/usr/bin/env bash

#Copyright (c) Facebook, Inc. and its affiliates.

#SBATCH --job-name=curiosity
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-medium
#SBATCH --chdir=/fs/clip-scratch/entilzha/curiosity
#SBATCH --output=/fs/www-users/entilzha/logs/%A.log
#SBATCH --error=/fs/www-users/entilzha/logs/%A.log
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10g
#SBATCH --partition=gpu
#SBATCH --exclude=materialgpu00

set -x
hostname
nvidia-smi
source /fs/clip-quiz/entilzha/anaconda3/etc/profile.d/conda.sh > /dev/null 2> /dev/null
conda activate curiosity
pwd
# $1 is the model name, eg "glove_bilstm"
# $2 is the random seed, eg 42
srun allennlp train --include-package curiosity -s models/${1}${2} -f configs/generated/${1}.json -o '{"trainer": {"cuda_device": 0}, "pytorch_seed": '${2}', "numpy_seed": '${2}', "random_seed": '${2}'}'
srun allennlp evaluate --include-package curiosity --output-file experiments/${1}${2}_val_metrics.json models/${1}${2} dialog_data/curiosity_dialogs.val.json
srun allennlp evaluate --include-package curiosity --output-file experiments/${1}${2}_test_metrics.json models/${1}${2} dialog_data/curiosity_dialogs.test.json
