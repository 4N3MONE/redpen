# Define abilities, epochs, and model names
abilities=("readability" "completeness" "grammar")
epochs=10
model_names=("bert-large-uncased" "bert-base-uncased" "microsoft/deberta-v3-base" "microsoft/deberta-v3-large" "roberta-base" "roberta-large")

# Iterate over configurations
for model_name in "${model_names[@]}"; do
  for ability in "${abilities[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python eval_baseline.py --ability "$ability" --epoch "$epochs" --model_name "$model_name"
  done
done
