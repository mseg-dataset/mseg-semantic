#!/bin/bash

export outf=2021_11_26_test
mkdir -p ${outf}


datasets=(
	wilddash-19
	camvid-11
	scannet-20
	kitti-19
	pascal-context-60
	voc2012
	
	ade20k-150
	bdd
	cityscapes-19
	coco-panoptic-133
	idd-39
	mapillary-public65
	sunrgbd-37
	)

training_datasets=(
	ade20k-150
	bdd
	cityscapes-19
	coco-panoptic-133
	idd-39
	mapillary-public65
	sunrgbd-37
	)

base_sizes=(
	360
	720
	1080
	#480
	#2160
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

relabeled_model_names=(
	mseg-3m
	mseg-3m-480p
	mseg-3m-720p
	mseg-1m
	)

for base_size in ${base_sizes[@]}; do
	for dataset in ${datasets[@]}; do
		for model_name in ${model_names[@]}; do
			
			d_folder=$dataset
			# check if eval dataset is one of the training datasets
			if [[ " ${training_datasets[@]} " =~ " ${dataset} " ]]; then
				d_folder=${d_folder}_universal

				# check if eval model is one of the relabeled models
				if [[ " ${relabeled_model_names[@]} " =~ " ${model_name} " ]]; then
					d_folder=${d_folder}_relabeled
				fi
			fi	

			# Mapillary has larger resolution images, and requires more GPU memory
			if [[ $dataset == *"mapillary"* ]]
			then
				script_name="eval_universal_tax_model_overcap_3gpu.sh"
			else
				script_name="eval_universal_tax_model_overcap_1gpu.sh"
			fi

			echo " "
			echo "Evaluate on: ${d_folder}"
			echo $script_name

			#sbatch --dependency=singleton --job-name=mseg_eval_A -c 5 -p short -x jarvis,vicki,cortana,gideon,ephemeral-3 --gres=gpu:1 \
			sbatch -c 5 --job-name=mseg_eval_overcap_A \
			-o ${outf}/${model_name}_${base_size}_${dataset}.log \
			${script_name} ${base_size} ${model_name} ${dataset} ${d_folder}

			echo " "
		done
	done
done
