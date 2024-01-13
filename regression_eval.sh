# #!/bin/bash

# # Define abilities, epochs, and model names
# abilities=("readability" "completeness" "grammar")
# model_names=("microsoft/deberta-v3-base" "microsoft/deberta-v3-large" "roberta-base" "roberta-large" "bert-base-uncased" "bert-large-uncased")

# # Iterate over configurations
# for model_name in "${model_names[@]}"; do
#   for ability in "${abilities[@]}"; do
#     CUDA_VISIBLE_DEVICES=1 python eval_regression.py --ability "$ability"  --model_name "$model_name"
#   done
# done


abilities="completeness"
model_names=("microsoft/deberta-v3-large" "roberta-large" "bert-large-uncased")

# Iterate over configurations
for model_name in "${model_names[@]}"; do
  for ability in "${abilities[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python eval_regression.py --ability "$ability"  --model_name "$model_name"
  done
done
