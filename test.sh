#!/bin/bash

export PYTHONPATH=$(pwd)

# evaluation example on all M1 task
METRIC_TYPE=M1  # choose from M1 to M5

# drd2
python evaluation.py --eval_file "data/drd2/test.txt" \
--load_model_dir saved/best_model/${METRIC_TYPE}/drd2/drd2.best_model \
--load_model_config_dir saved/best_model/${METRIC_TYPE}/drd2/model_config.json \
--task_tag drd2 \
--metric_type ${METRIC_TYPE} \
--device 0

# qed
python evaluation.py --eval_file "data/qed/test.txt" \
--load_model_dir saved/best_model/${METRIC_TYPE}/qed/qed.best_model \
--load_model_config_dir saved/best_model/${METRIC_TYPE}/qed/model_config.json \
--task_tag qed \
--metric_type ${METRIC_TYPE} \
--device 0

# logp04
python evaluation.py --eval_file "data/logp04/test.txt" \
--load_model_dir saved/best_model/${METRIC_TYPE}/logp04/logp04.best_model \
--load_model_config_dir saved/best_model/${METRIC_TYPE}/logp04/model_config.json \
--task_tag logp04 \
--metric_type ${METRIC_TYPE} \
--device 0

# logp06
python evaluation.py --eval_file "data/logp06/test.txt" \
--load_model_dir saved/best_model/${METRIC_TYPE}/logp06/logp06.best_model \
--load_model_config_dir saved/best_model/${METRIC_TYPE}/logp06/model_config.json \
--task_tag logp06 \
--metric_type ${METRIC_TYPE} \
--device 0

# GuacaMol_qed
python evaluation.py --eval_file "data/GuacaMol_qed/test.txt" \
--load_model_dir saved/best_model/${METRIC_TYPE}/GuacaMol_qed/GuacaMol_qed.best_model \
--load_model_config_dir saved/best_model/${METRIC_TYPE}/GuacaMol_qed/model_config.json \
--task_tag GuacaMol_qed \
--metric_type ${METRIC_TYPE} \
--device 0

# Moses_qed
python evaluation.py --eval_file "data/Moses_qed/test.txt" \
--load_model_dir saved/best_model/${METRIC_TYPE}/Moses_qed/Moses_qed.best_model \
--load_model_config_dir saved/best_model/${METRIC_TYPE}/Moses_qed/model_config.json \
--task_tag Moses_qed \
--metric_type ${METRIC_TYPE} \
--device 0

# reverse_logp04
python evaluation.py --eval_file "data/reverse_logp04/test.txt" \
--load_model_dir saved/best_model/${METRIC_TYPE}/reverse_logp04/reverse_logp04.best_model \
--load_model_config_dir saved/best_model/${METRIC_TYPE}/reverse_logp04/model_config.json \
--task_tag reverse_logp04 \
--metric_type ${METRIC_TYPE} \
--device 0
