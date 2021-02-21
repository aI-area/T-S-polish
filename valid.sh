#!/bin/bash

export PYTHONPATH=$(pwd)

#######################################
### SETTING YOUR OWN PARAMETER HEAR ###
TASK_TAG=qed # qed, drd2, logp04, logp06, GuacaMol_qed, Moses_qed, GuacaMol_multi_prop, reverse_logp04
METRIC_TYPE=M1 # M1, M2, M3, M4, M5, M6
MODEL_DIR=saved/model/${TASK_TAG}
BEGIN_NUM=1
END_NUM=30
DEVICE=1
#######################################

echo "Valid file: data/${TASK_TAG}/valid.txt"

for ((i=${BEGIN_NUM}; i<=${END_NUM}; i++)); do
    f=${MODEL_DIR}/model.iter-$i
    if [ -e $f ]; then
        echo "============================================================"
        echo "${f}"
        python evaluation.py --eval_file "data/${TASK_TAG}/valid.txt" \
        --load_model_dir "${f}" \
        --load_model_config_dir "${MODEL_DIR}/model_config.json" \
        --task_tag "${TASK_TAG}" \
        --metric_type "${METRIC_TYPE}" \
        --device "${DEVICE}"
        echo "============================================================"
    fi
done
echo "Done."

