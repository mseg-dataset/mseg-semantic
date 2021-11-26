#!/bin/bash

export outf=2021_11_27_test
mkdir -p ${outf}


datasets=(
	wilddash-19
	camvid-11
	scannet-20
	kitti-19
	pascal-context-60
	voc2012
	)

base_sizes=(
	360
	720
	1080
	# 480
	# 2160
	)

model_name="mseg-naive-baseline-1m"


for base_size in ${base_sizes[@]}; do
	for ((i=0;i<${#datasets[@]};++i)); do
		dataset="${datasets[i]}"
		sbatch -c 5 --job-name=mseg_eval_overcap_A \
		-o ${outf}/${model_name}_${base_size}_${dataset}.log \
		eval_naive_tax_model_overcap.sh ${base_size} ${model_name} ${dataset}
	done
done

