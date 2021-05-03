export outf=0329_halfway
mkdir ${outf}

# 8571-8580
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/coco-v1 tool/train-qvga-one-copy.sh  1080/single_universal.yaml False exp ${WORK}/copies/final_train/1080-halfway/coco-panoptic-v1-sr coco-panoptic-v1-sr
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/ade-v1 tool/train-qvga-one-copy.sh  1080/single_universal.yaml False exp ${WORK}/copies/final_train/1080-halfway/ade20k-v1-sr ade20k-v1-sr
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/idd-new tool/train-qvga-one-copy.sh  1080/single_universal.yaml False exp ${WORK}/copies/final_train/1080-halfway/idd-new idd-new
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/sunrgbd-37 tool/train-qvga-one-copy.sh  1080/single_universal.yaml False exp ${WORK}/copies/final_train/1080-halfway/sunrgbd-37-sr sunrgbd-37-sr
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/bdd tool/train-qvga-one-copy.sh  1080/single_universal.yaml False exp ${WORK}/copies/final_train/1080-halfway/bdd-sr bdd-sr
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/cityscapes tool/train-qvga-one-copy.sh  1080/single_universal.yaml False exp ${WORK}/copies/final_train/1080-halfway/cityscapes cityscapes
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/mapillary tool/train-qvga-one-copy.sh  1080/single_universal.yaml False exp ${WORK}/copies/final_train/1080-halfway/mapillary mapillary

# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/scannet-20 tool/train-qvga-one-copy.sh  1080/single.yaml False exp ${WORK}/copies/final_train/1080-halfway/scannet-20 scannet-20
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/camvid tool/train-qvga-one-copy.sh  1080/single.yaml False exp ${WORK}/copies/final_train/1080-halfway/camvid camvid
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/voc2012 tool/train-qvga-one-copy.sh  1080/single.yaml False exp ${WORK}/copies/final_train/1080-halfway/voc2012 voc2012
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/kitti tool/train-qvga-one-copy.sh  1080/single.yaml False exp ${WORK}/copies/final_train/1080-halfway/kitti kitti
sbatch -p quadro --gres=gpu:8   -c 80 -t 2-00:00:00 -o ${outf}/pascal-context tool/train-qvga-one-copy.sh  1080/single.yaml False exp ${WORK}/copies/final_train/1080-halfway/pascal-context-60 pascal-context-60
# 9888




# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/cityscapes-v2 tool/train-qvga-one-copy.sh  1080/single_universal.yaml False exp ${WORK}/copies/final_train/1080-halfway/cit-v2 cityscapes-v2
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/cityscapes tool/train-qvga-one-copy.sh  1080/single_universal.yaml False exp ${WORK}/copies/final_train/1080-halfway/cityscapes cityscapes
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/kitti-sr tool/train-qvga-mix-copy.sh  1080/kitti-sr.yaml False exp ${WORK}/copies/final_train/1080-halfway-1/kitti-sr


# 7075-7077
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/kitti-sr tool/train-qvga-mix-copy.sh  1080/kitti-sr.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/kitti-sr
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/camvid-sr tool/train-qvga-mix-copy.sh  1080/camvid-sr.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/camvid-sr
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/voc2012-sr tool/train-qvga-mix-copy.sh  1080/voc2012-sr.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/voc2012-sr




# tool/train-qvga-mix-copy.sh  1080/kitti.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/test


# 6920-6922
# 7100-7102 

# 7150-7251 now, gpu19
# 7254-55



# 6294-6296
# sbatch -p quadro --gres=gpu:8   -w isl-gpu3   -c 80 -t 2-00:00:00 -o ${outf}/mseg-stupid tool/train-qvga-mix-copy.sh  1080/mseg-stupid.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/mseg-stupid
# 7252-7253, gpu18


# 7256-7257
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu18   -c 80 -t 2-00:00:00 -o ${outf}/mseg-stupid-1 tool/train-qvga-mix-cd.sh  1080/mseg-stupid.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/mseg-stupid
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu18   -c 80 -t 2-00:00:00 -o ${outf}/mseg-stupid-2 tool/train-qvga-mix-cd.sh  1080/mseg-stupid.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/mseg-stupid

# 7410-12
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu19   -c 80 -t 2-00:00:00 -o ${outf}/mseg-unrelabeled tool/train-qvga-mix-copy.sh  1080/mseg-unrelabeled.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/mseg-unrelabeled
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu19   -c 80 -t 2-00:00:00 -o ${outf}/mseg-unrelabeled-1 tool/train-qvga-mix-cd.sh  1080/mseg-unrelabeled.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/mseg-unrelabeled
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu19   -c 80 -t 2-00:00:00 -o ${outf}/mseg-unrelabeled-2 tool/train-qvga-mix-cd.sh  1080/mseg-unrelabeled.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/mseg-unrelabeled


# 7419, gpu4


# 7436-7453

# 7972-7991

# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu25   -c 80 -t 2-00:00:00 -o ${outf}/mseg-3m tool/train-qvga-mix-copy.sh  1080/mseg-3m.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/mseg-3m
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu25   -c 80 -t 2-00:00:00 -o ${outf}/mseg-3m-1 tool/train-qvga-mix-cd.sh  1080/mseg-3m.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/mseg-3m
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu25   -c 80 -t 2-00:00:00 -o ${outf}/mseg-3m-2 tool/train-qvga-mix-cd.sh  1080/mseg-3m.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/mseg-3m
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu25   -c 80 -t 2-00:00:00 -o ${outf}/mseg-3m-3 tool/train-qvga-mix-cd.sh  1080/mseg-3m.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/mseg-3m
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu25   -c 80 -t 2-00:00:00 -o ${outf}/mseg-3m-4 tool/train-qvga-mix-cd.sh  1080/mseg-3m.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/mseg-3m
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu25   -c 80 -t 2-00:00:00 -o ${outf}/mseg-3m-5 tool/train-qvga-mix-cd.sh  1080/mseg-3m.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/mseg-3m
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu25   -c 80 -t 2-00:00:00 -o ${outf}/mseg-3m-6 tool/train-qvga-mix-cd.sh  1080/mseg-3m.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/mseg-3m
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu25   -c 80 -t 2-00:00:00 -o ${outf}/mseg-3m-7 tool/train-qvga-mix-cd.sh  1080/mseg-3m.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/mseg-3m


# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu24   -c 80 -t 2-00:00:00 -o ${outf}/mseg tool/train-qvga-mix-copy.sh  1080/mseg.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/mseg
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu24   -c 80 -t 2-00:00:00 -o ${outf}/mseg-1 tool/train-qvga-mix-cd.sh  1080/mseg.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/mseg
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu24   -c 80 -t 2-00:00:00 -o ${outf}/mseg-2 tool/train-qvga-mix-cd.sh  1080/mseg.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/mseg


# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu19   -c 80 -t 2-00:00:00 -o ${outf}/mseg-unrelabeled tool/train-qvga-mix-copy.sh  1080/mseg-unrelabeled.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/mseg-unrelabeled
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu19   -c 80 -t 2-00:00:00 -o ${outf}/mseg-unrelabeled-1 tool/train-qvga-mix-cd.sh  1080/mseg-unrelabeled.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/mseg-unrelabeled
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu19   -c 80 -t 2-00:00:00 -o ${outf}/mseg-unrelabeled-2 tool/train-qvga-mix-cd.sh  1080/mseg-unrelabeled.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/mseg-unrelabeled


# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu3   -c 80 -t 2-00:00:00 -o ${outf}/mseg-lowres tool/train-qvga-mix-copy.sh  1080/mseg-lowres.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/mseg-lowres
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu3   -c 80 -t 2-00:00:00 -o ${outf}/mseg-lowres-1 tool/train-qvga-mix-cd.sh  1080/mseg-lowres.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/mseg-lowres
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu3   -c 80 -t 2-00:00:00 -o ${outf}/mseg-lowres-2 tool/train-qvga-mix-cd.sh  1080/mseg-lowres.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/mseg-lowres

# sbatch -p quadro --qos=normal --gres=gpu:8 -c 80 -t 2-00:00:00 -o ${outf}/sunrgbd-37-sr-new tool/train-qvga-mix-copy.sh  1080/sunrgbd-37-sr.yaml False exp ${WORK}/copies/final_train/1080-halfway-1-new/sunrgbd-37-sr


# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu4   -c 80 -t 2:20:00 -o ${outf}/mseg-lowres-test tool/train-qvga-mix-copy.sh  1080/mseg-lowres.yaml False exp ${WORK}/copies/test-new
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu4   -c 80 -t 2:20:00 -o ${outf}/mseg-lowres-1-test tool/train-qvga-mix-cd.sh  1080/mseg-lowres.yaml False exp ${WORK}/copies/test-new
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu4   -c 80 -t 2:20:00 -o ${outf}/mseg-lowres-2-test tool/train-qvga-mix-cd.sh  1080/mseg-lowres.yaml False exp ${WORK}/copies/test-new







