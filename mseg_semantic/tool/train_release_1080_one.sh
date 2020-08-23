export outf=0424_release/
mkdir ${outf}

# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/scannet-20 tool/train-qvga-one-copy.sh  1080_release/single.yaml False exp ${WORK}/copies/final_train/1080_release/scannet-20 scannet-20
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/camvid-11 tool/train-qvga-one-copy.sh  1080_release/single.yaml False exp ${WORK}/copies/final_train/1080_release/camvid-11 camvid-11
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/voc2012 tool/train-qvga-one-copy.sh  1080_release/single.yaml False exp ${WORK}/copies/final_train/1080_release/voc2012 voc2012
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/kitti-19 tool/train-qvga-one-copy.sh  1080_release/single.yaml False exp ${WORK}/copies/final_train/1080_release/kitti-19 kitti-19
# sbatch -p quadro --gres=gpu:8   	 -c 80 -t 2-00:00:00 -o ${outf}/pascal-context-60 tool/train-qvga-one-copy.sh  1080_release/single.yaml False exp ${WORK}/copies/final_train/1080_release/pascal-context-60 pascal-context-60

# 14483-86
sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/coco-panoptic-133 tool/train-qvga-one-copy.sh  1080_release/single_universal.yaml False exp ${WORK}/copies/final_train/1080_release/coco-panoptic-133 coco-panoptic-133
sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/ade20k-150 tool/train-qvga-one-copy.sh  1080_release/single_universal.yaml False exp ${WORK}/copies/final_train/1080_release/ade20k-150 ade20k-150
sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/sunrgbd-37 tool/train-qvga-one-copy.sh  1080_release/single_universal.yaml False exp ${WORK}/copies/final_train/1080_release/sunrgbd-37 sunrgbd-37
sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/bdd tool/train-qvga-one-copy.sh  1080_release/single_universal.yaml False exp ${WORK}/copies/final_train/1080_release/bdd bdd
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/idd-39 tool/train-qvga-one-copy.sh  1080_release/single_universal.yaml False exp ${WORK}/copies/final_train/1080_release/idd-39 idd-39
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/cityscapes-19 tool/train-qvga-one-copy.sh  1080_release/single_universal.yaml False exp ${WORK}/copies/final_train/1080_release/cityscapes-19 cityscapes-19
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/mapillary-public65 tool/train-qvga-one-copy.sh  1080_release/single_universal.yaml False exp ${WORK}/copies/final_train/1080_release/mapillary-public65 mapillary-public65


