#!/usr/bin/env bash
# Experiments for: glove_bilstm
allennlp train --include-package curiosity -s models/glove_bilstm -f configs/generated/glove_bilstm.json -o '{"trainer": {"cuda_device": 0}}'
allennlp evaluate --include-package curiosity --output-file experiments/glove_bilstm_val_metrics.json models/glove_bilstm dialog_data/curiosity_dialogs.val.json
allennlp evaluate --include-package curiosity --output-file experiments/glove_bilstm_test_metrics.json models/glove_bilstm dialog_data/curiosity_dialogs.test.json
allennlp evaluate --include-package curiosity --output-file experiments/glove_bilstm_zero_metrics.json models/glove_bilstm dialog_data/curiosity_dialogs.test_zero.json

# Experiments for: glove_bilstm-known
allennlp train --include-package curiosity -s models/glove_bilstm-known -f configs/generated/glove_bilstm-known.json -o '{"trainer": {"cuda_device": 0}}'
allennlp evaluate --include-package curiosity --output-file experiments/glove_bilstm-known_val_metrics.json models/glove_bilstm-known dialog_data/curiosity_dialogs.val.json
allennlp evaluate --include-package curiosity --output-file experiments/glove_bilstm-known_test_metrics.json models/glove_bilstm-known dialog_data/curiosity_dialogs.test.json
allennlp evaluate --include-package curiosity --output-file experiments/glove_bilstm-known_zero_metrics.json models/glove_bilstm-known dialog_data/curiosity_dialogs.test_zero.json

# Experiments for: glove_bilstm-da
allennlp train --include-package curiosity -s models/glove_bilstm-da -f configs/generated/glove_bilstm-da.json -o '{"trainer": {"cuda_device": 0}}'
allennlp evaluate --include-package curiosity --output-file experiments/glove_bilstm-da_val_metrics.json models/glove_bilstm-da dialog_data/curiosity_dialogs.val.json
allennlp evaluate --include-package curiosity --output-file experiments/glove_bilstm-da_test_metrics.json models/glove_bilstm-da dialog_data/curiosity_dialogs.test.json
allennlp evaluate --include-package curiosity --output-file experiments/glove_bilstm-da_zero_metrics.json models/glove_bilstm-da dialog_data/curiosity_dialogs.test_zero.json

# Experiments for: glove_bilstm-like
allennlp train --include-package curiosity -s models/glove_bilstm-like -f configs/generated/glove_bilstm-like.json -o '{"trainer": {"cuda_device": 0}}'
allennlp evaluate --include-package curiosity --output-file experiments/glove_bilstm-like_val_metrics.json models/glove_bilstm-like dialog_data/curiosity_dialogs.val.json
allennlp evaluate --include-package curiosity --output-file experiments/glove_bilstm-like_test_metrics.json models/glove_bilstm-like dialog_data/curiosity_dialogs.test.json
allennlp evaluate --include-package curiosity --output-file experiments/glove_bilstm-like_zero_metrics.json models/glove_bilstm-like dialog_data/curiosity_dialogs.test_zero.json

# Experiments for: glove_bilstm-facts
allennlp train --include-package curiosity -s models/glove_bilstm-facts -f configs/generated/glove_bilstm-facts.json -o '{"trainer": {"cuda_device": 0}}'
allennlp evaluate --include-package curiosity --output-file experiments/glove_bilstm-facts_val_metrics.json models/glove_bilstm-facts dialog_data/curiosity_dialogs.val.json
allennlp evaluate --include-package curiosity --output-file experiments/glove_bilstm-facts_test_metrics.json models/glove_bilstm-facts dialog_data/curiosity_dialogs.test.json
allennlp evaluate --include-package curiosity --output-file experiments/glove_bilstm-facts_zero_metrics.json models/glove_bilstm-facts dialog_data/curiosity_dialogs.test_zero.json

# Experiments for: bert
allennlp train --include-package curiosity -s models/bert -f configs/generated/bert.json -o '{"trainer": {"cuda_device": 0}}'
allennlp evaluate --include-package curiosity --output-file experiments/bert_val_metrics.json models/bert dialog_data/curiosity_dialogs.val.json
allennlp evaluate --include-package curiosity --output-file experiments/bert_test_metrics.json models/bert dialog_data/curiosity_dialogs.test.json
allennlp evaluate --include-package curiosity --output-file experiments/bert_zero_metrics.json models/bert dialog_data/curiosity_dialogs.test_zero.json

# Experiments for: bert-known
allennlp train --include-package curiosity -s models/bert-known -f configs/generated/bert-known.json -o '{"trainer": {"cuda_device": 0}}'
allennlp evaluate --include-package curiosity --output-file experiments/bert-known_val_metrics.json models/bert-known dialog_data/curiosity_dialogs.val.json
allennlp evaluate --include-package curiosity --output-file experiments/bert-known_test_metrics.json models/bert-known dialog_data/curiosity_dialogs.test.json
allennlp evaluate --include-package curiosity --output-file experiments/bert-known_zero_metrics.json models/bert-known dialog_data/curiosity_dialogs.test_zero.json

# Experiments for: bert-da
allennlp train --include-package curiosity -s models/bert-da -f configs/generated/bert-da.json -o '{"trainer": {"cuda_device": 0}}'
allennlp evaluate --include-package curiosity --output-file experiments/bert-da_val_metrics.json models/bert-da dialog_data/curiosity_dialogs.val.json
allennlp evaluate --include-package curiosity --output-file experiments/bert-da_test_metrics.json models/bert-da dialog_data/curiosity_dialogs.test.json
allennlp evaluate --include-package curiosity --output-file experiments/bert-da_zero_metrics.json models/bert-da dialog_data/curiosity_dialogs.test_zero.json

# Experiments for: bert-like
allennlp train --include-package curiosity -s models/bert-like -f configs/generated/bert-like.json -o '{"trainer": {"cuda_device": 0}}'
allennlp evaluate --include-package curiosity --output-file experiments/bert-like_val_metrics.json models/bert-like dialog_data/curiosity_dialogs.val.json
allennlp evaluate --include-package curiosity --output-file experiments/bert-like_test_metrics.json models/bert-like dialog_data/curiosity_dialogs.test.json
allennlp evaluate --include-package curiosity --output-file experiments/bert-like_zero_metrics.json models/bert-like dialog_data/curiosity_dialogs.test_zero.json

# Experiments for: bert-facts
allennlp train --include-package curiosity -s models/bert-facts -f configs/generated/bert-facts.json -o '{"trainer": {"cuda_device": 0}}'
allennlp evaluate --include-package curiosity --output-file experiments/bert-facts_val_metrics.json models/bert-facts dialog_data/curiosity_dialogs.val.json
allennlp evaluate --include-package curiosity --output-file experiments/bert-facts_test_metrics.json models/bert-facts dialog_data/curiosity_dialogs.test.json
allennlp evaluate --include-package curiosity --output-file experiments/bert-facts_zero_metrics.json models/bert-facts dialog_data/curiosity_dialogs.test_zero.json
