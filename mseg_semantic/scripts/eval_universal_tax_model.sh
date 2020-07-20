#!/bin/bash

base_size=$1
model_name=$2
dataset_name=$3

config_fpath=../config/test/default_config_${base_size}_ss.yaml
model_fpath=../../../pretrained-semantic-models/${model_name}/${model_name}.pth

results_path=../../../pretrained-semantic-models/${model_name}/${model_name}/${dataset_name}/${base_size}/results.txt
if [ -f ${results_path} ]
then
    echo "Results file already exists. Quitting"
    exit
else
    echo "Results file not found. Executing..."
fi

echo "On node ${HOSTNAME}"
echo "Running model ${model_name} on dataset ${dataset_name}, using model at ${model_fpath}"

python -u ../tool/test_universal_tax.py --config=${config_fpath} dataset ${dataset_name} model_path ${model_fpath} model_name ${model_name}
