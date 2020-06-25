#!/bin/bash

export outf=2020_06_24_test
mkdir -p ${outf}


datasets=(
	wilddash-19
	camvid-11
	scannet-20
	kitti-19
	pascal-context-60
	voc2012
	
	#ade20k-150
	#bdd
	#cityscapes-19
	#coco-panoptic-133
	#idd-39
	#mapillary-public65
	#sunrgbd-37
	)

base_sizes=(
	360
	720
	#1080
	)

model_names=(
	ade20k-150-1m
	bdd-1m
	cityscapes-19-1m
	coco-panoptic-133-1m
	idd-39-1m
	mseg-3m-480p
	mapillary-65-1m
	mseg-3m-720p
	mseg-1m
	mseg-mgda-1m
	mseg-3m
	sunrgbd-37-1m
	mseg-unrelabeled-1m	
	)

for base_size in ${base_sizes[@]}; do
	for dataset in ${datasets[@]}; do
		for model_name in ${model_names[@]}; do

			sbatch --qos=overcap -c 5 -p short -x jarvis --gres=gpu:1 -o ${outf}/${model_name}_${base_size}_${dataset}.log eval_universal_tax_model.sh \
			${base_size} ${model_name} ${dataset}
		done
	done
done
