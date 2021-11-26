#!/bin/bash

export outf=2021_11_26_test
mkdir -p ${outf}


datasets=(
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

o_model_names=(
	camvid-11-1m
	scannet-20-1m
	kitti-19-1m
	pascal-context-60-1m
	voc2012-1m
	)


for base_size in ${base_sizes[@]}; do
	for ((i=0;i<${#datasets[@]};++i)); do
		dataset="${datasets[i]}"
		model_name="${o_model_names[i]}"
		sbatch -c 5 --job-name=mseg_eval_overcap_A \
		-o ${outf}/${model_name}_${base_size}_${dataset}.log \
		eval_oracle_tax_model_overcap.sh ${base_size} ${model_name} ${dataset}
	done
done

