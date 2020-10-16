# Copyright (c) Facebook, Inc. and its affiliates.

# This assumes the the curiosity conda env is set
# Run for example:
# bash run_local.sh bert-facts 1
# bash run_local.sh bert-facts 2


allennlp train --include-package curiosity -s models/${1}${2} -f configs/generated/${1}.json -o '{"trainer": {"cuda_device": 0}, "pytorch_seed": '${2}', "numpy_seed": '${2}', "random_seed": '${2}'}'
allennlp evaluate --include-package curiosity --output-file experiments/${1}${2}_val_metrics.json models/${1}${2} dialog_data/curiosity_dialogs.val.json
allennlp evaluate --include-package curiosity --output-file experiments/${1}${2}_test_metrics.json models/${1}${2} dialog_data/curiosity_dialogs.test.json
