#!/bin/bash

# Define abilities, epochs, and model names
# abilities=("readability" "completeness" "grammar")
# epochs=50
# model_names=("microsoft/deberta-v3-base" "microsoft/deberta-v3-large" "roberta-base" "roberta-large" "bert-base-uncased" "bert-large-uncased")

# # Iterate over configurations
# for model_name in "${model_names[@]}"; do
#   for ability in "${abilities[@]}"; do
#     CUDA_VISIBLE_DEVICES=1 python train_regression.py --ability "$ability" --epoch "$epochs" --model_name "$model_name"
#   done
# done

abilities="completeness"
epochs=10
model_names=("microsoft/deberta-v3-large" "roberta-large" "bert-large-uncased")

# Iterate over configurations
for model_name in "${model_names[@]}"; do
  for ability in "${abilities[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python train_regression.py --ability "$ability" --epoch "$epochs" --model_name "$model_name"
  done
done

