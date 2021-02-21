#!/bin/bash

export PYTHONPATH=$(pwd)

TASK_TAG=qed # qed, drd2, logp04, logp06, GuacaMol_qed, Moses_qed, GuacaMol_multi_prop, reverse_logp04

python train.py \
--train_file "data/${TASK_TAG}/train_processed" \
--vocab "data/${TASK_TAG}/vocab.txt" \
--model_save_dir "saved/model" \
--hidden_size 300 \
--epoch 30 --lr 0.0001 \
--anneal_rate 0.9 --anneal_interval 1 \
--batch_size 32 \
--task_tag "${TASK_TAG}" \
--share_embedding \
--device 6
echo "Done."
