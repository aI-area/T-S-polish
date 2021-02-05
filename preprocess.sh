#!/bin/bash

export PYTHONPATH=$(pwd)

DATASET=("drd2" "qed" "logp04" "logp06" "GuacaMol_qed" "Moses_qed" "GuacaMol_multi_prop" "reverse_logp04")
N_CPU=35

for dataset_name in "${DATASET[@]}"; do
    echo "==========================================="
    echo "Process data: ${dataset_name}"
    dataset_dir="data/${dataset_name}"
    echo "Generate train score data..."
    python scripts/count_center_score.py --smiles_pair_file "${dataset_dir}/train_pairs.txt" --result_file "${dataset_dir}/train_score_data.txt" --ncpu ${N_CPU}
    echo "Preprocess train file..."
    python scripts/preprocess.py --data_file "${dataset_dir}/train_score_data.txt" --save_dir "${dataset_dir}/train_processed" --ncpu ${N_CPU}
    echo "Process ${dataset_name} dataset Done."
done
