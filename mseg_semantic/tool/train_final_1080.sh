export outf=0327-fixedbug
mkdir ${outf}

# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/bdd tool/train-qvga-mix-copy.sh  1080/bdd.yaml False exp ${WORK}/copies/final_train/1080/bdd
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/bdd tool/train-qvga-mix-copy.sh  1080/bdd.yaml False exp ${WORK}/copies/final_train/1080/bdd

# 6892-6894
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/coco-panoptic-v1-sr tool/train-qvga-mix-copy.sh  1080/coco-panoptic-v1-sr.yaml False exp ${WORK}/copies/final_train/1080-1-new/coco-panoptic-v1-sr
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/bdd-sr tool/train-qvga-mix-copy.sh  1080/bdd-sr.yaml False exp ${WORK}/copies/final_train/1080-1-new/bdd-sr
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/ade20k-v1-sr tool/train-qvga-mix-copy.sh  1080/ade20k-v1-sr.yaml False exp ${WORK}/copies/final_train/1080-1-new/ade20k-v1-sr

# after john made changes to ade20k taxonomy, 7091
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/ade20k-v1-sr tool/train-qvga-mix-copy.sh  1080/ade20k-v1-sr.yaml False exp ${WORK}/copies/final_train/1080-1-new/ade20k-v1-sr


# 6927 - 6929
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/sunrgbd-37-sr tool/train-qvga-mix-copy.sh  1080/sunrgbd-37-sr.yaml False exp ${WORK}/copies/final_train/1080-1-new/sunrgbd-37-sr
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/idd-new tool/train-qvga-mix-copy.sh  1080/idd-new.yaml False exp ${WORK}/copies/final_train/1080-1-new/idd-new
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/cityscapes tool/train-qvga-mix-copy.sh  1080/cityscapes.yaml False exp ${WORK}/copies/final_train/1080-1-new/cityscapes

# 6999- 7002
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/mapillary tool/train-qvga-mix-copy.sh  1080/mapillary.yaml False exp ${WORK}/copies/final_train/1080-1-new/mapillary
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/voc2012 tool/train-qvga-mix-copy.sh  1080/voc2012.yaml False exp ${WORK}/copies/final_train/1080-1-new/voc2012
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/scannet-20 tool/train-qvga-mix-copy.sh  1080/scannet-20.yaml False exp ${WORK}/copies/final_train/1080-1-new/scannet-20
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/camvid tool/train-qvga-mix-copy.sh  1080/camvid.yaml False exp ${WORK}/copies/final_train/1080-1-new/camvid

# 7051
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/kitti tool/train-qvga-mix-copy.sh  1080/kitti.yaml False exp ${WORK}/copies/final_train/1080-1/kitti
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/kitti-sr tool/train-qvga-mix-copy.sh  1080/kitti-sr.yaml False exp ${WORK}/copies/final_train/1080-1/kitti-sr


# 7075-7077
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/kitti-sr tool/train-qvga-mix-copy.sh  1080/kitti-sr.yaml False exp ${WORK}/copies/final_train/1080-1-new/kitti-sr
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/camvid-sr tool/train-qvga-mix-copy.sh  1080/camvid-sr.yaml False exp ${WORK}/copies/final_train/1080-1-new/camvid-sr
# sbatch -p quadro --gres=gpu:8        -c 80 -t 2-00:00:00 -o ${outf}/voc2012-sr tool/train-qvga-mix-copy.sh  1080/voc2012-sr.yaml False exp ${WORK}/copies/final_train/1080-1-new/voc2012-sr




# tool/train-qvga-mix-copy.sh  1080/kitti.yaml False exp ${WORK}/copies/final_train/1080-1-new/test


# 6920-6922
# 7100-7102 

# 7150-7251 now, gpu19
# 7254-55



# 6294-6296
# sbatch -p quadro --gres=gpu:8   -w isl-gpu3   -c 80 -t 2-00:00:00 -o ${outf}/mseg-stupid tool/train-qvga-mix-copy.sh  1080/mseg-stupid.yaml False exp ${WORK}/copies/final_train/1080-1-new/mseg-stupid
# 7252-7253, gpu18


# 7256-7257
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu18   -c 80 -t 2-00:00:00 -o ${outf}/mseg-stupid-1 tool/train-qvga-mix-cd.sh  1080/mseg-stupid.yaml False exp ${WORK}/copies/final_train/1080-1-new/mseg-stupid
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu18   -c 80 -t 2-00:00:00 -o ${outf}/mseg-stupid-2 tool/train-qvga-mix-cd.sh  1080/mseg-stupid.yaml False exp ${WORK}/copies/final_train/1080-1-new/mseg-stupid

# 7410-12
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu19   -c 80 -t 2-00:00:00 -o ${outf}/mseg-unrelabeled tool/train-qvga-mix-copy.sh  1080/mseg-unrelabeled.yaml False exp ${WORK}/copies/final_train/1080-1-new/mseg-unrelabeled
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu19   -c 80 -t 2-00:00:00 -o ${outf}/mseg-unrelabeled-1 tool/train-qvga-mix-cd.sh  1080/mseg-unrelabeled.yaml False exp ${WORK}/copies/final_train/1080-1-new/mseg-unrelabeled
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu19   -c 80 -t 2-00:00:00 -o ${outf}/mseg-unrelabeled-2 tool/train-qvga-mix-cd.sh  1080/mseg-unrelabeled.yaml False exp ${WORK}/copies/final_train/1080-1-new/mseg-unrelabeled


# 7419, gpu4


# 7436-7453

# 7972-7991

# 8256-8269 -fixed bug
sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu24   -c 80 -t 2-00:00:00 -o ${outf}/mseg-3m tool/train-qvga-mix-copy.sh  1080/mseg-3m.yaml False exp ${WORK}/copies/final_train/1080-1-new/mseg-3m
sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu24   -c 80 -t 2-00:00:00 -o ${outf}/mseg-3m-1 tool/train-qvga-mix-cd.sh  1080/mseg-3m.yaml False exp ${WORK}/copies/final_train/1080-1-new/mseg-3m
sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu24   -c 80 -t 2-00:00:00 -o ${outf}/mseg-3m-2 tool/train-qvga-mix-cd.sh  1080/mseg-3m.yaml False exp ${WORK}/copies/final_train/1080-1-new/mseg-3m
sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu24   -c 80 -t 2-00:00:00 -o ${outf}/mseg-3m-3 tool/train-qvga-mix-cd.sh  1080/mseg-3m.yaml False exp ${WORK}/copies/final_train/1080-1-new/mseg-3m
sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu24   -c 80 -t 2-00:00:00 -o ${outf}/mseg-3m-4 tool/train-qvga-mix-cd.sh  1080/mseg-3m.yaml False exp ${WORK}/copies/final_train/1080-1-new/mseg-3m
sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu24   -c 80 -t 2-00:00:00 -o ${outf}/mseg-3m-5 tool/train-qvga-mix-cd.sh  1080/mseg-3m.yaml False exp ${WORK}/copies/final_train/1080-1-new/mseg-3m
sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu24   -c 80 -t 2-00:00:00 -o ${outf}/mseg-3m-6 tool/train-qvga-mix-cd.sh  1080/mseg-3m.yaml False exp ${WORK}/copies/final_train/1080-1-new/mseg-3m
sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu24   -c 80 -t 2-00:00:00 -o ${outf}/mseg-3m-7 tool/train-qvga-mix-cd.sh  1080/mseg-3m.yaml False exp ${WORK}/copies/final_train/1080-1-new/mseg-3m


sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu25   -c 80 -t 2-00:00:00 -o ${outf}/mseg tool/train-qvga-mix-copy.sh  1080/mseg.yaml False exp ${WORK}/copies/final_train/1080-1-new/mseg
sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu25   -c 80 -t 2-00:00:00 -o ${outf}/mseg-1 tool/train-qvga-mix-cd.sh  1080/mseg.yaml False exp ${WORK}/copies/final_train/1080-1-new/mseg
sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu25   -c 80 -t 2-00:00:00 -o ${outf}/mseg-2 tool/train-qvga-mix-cd.sh  1080/mseg.yaml False exp ${WORK}/copies/final_train/1080-1-new/mseg


# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu19   -c 80 -t 2-00:00:00 -o ${outf}/mseg-unrelabeled tool/train-qvga-mix-copy.sh  1080/mseg-unrelabeled.yaml False exp ${WORK}/copies/final_train/1080-1-new/mseg-unrelabeled
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu19   -c 80 -t 2-00:00:00 -o ${outf}/mseg-unrelabeled-1 tool/train-qvga-mix-cd.sh  1080/mseg-unrelabeled.yaml False exp ${WORK}/copies/final_train/1080-1-new/mseg-unrelabeled
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu19   -c 80 -t 2-00:00:00 -o ${outf}/mseg-unrelabeled-2 tool/train-qvga-mix-cd.sh  1080/mseg-unrelabeled.yaml False exp ${WORK}/copies/final_train/1080-1-new/mseg-unrelabeled


sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu18   -c 80 -t 2-00:00:00 -o ${outf}/mseg-lowres tool/train-qvga-mix-copy.sh  1080/mseg-lowres.yaml False exp ${WORK}/copies/final_train/1080-1-new/mseg-lowres
sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu18   -c 80 -t 2-00:00:00 -o ${outf}/mseg-lowres-1 tool/train-qvga-mix-cd.sh  1080/mseg-lowres.yaml False exp ${WORK}/copies/final_train/1080-1-new/mseg-lowres
sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu18   -c 80 -t 2-00:00:00 -o ${outf}/mseg-lowres-2 tool/train-qvga-mix-cd.sh  1080/mseg-lowres.yaml False exp ${WORK}/copies/final_train/1080-1-new/mseg-lowres

# sbatch -p quadro --qos=normal --gres=gpu:8 -c 80 -t 2-00:00:00 -o ${outf}/sunrgbd-37-sr-new tool/train-qvga-mix-copy.sh  1080/sunrgbd-37-sr.yaml False exp ${WORK}/copies/final_train/1080-1-new/sunrgbd-37-sr


# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu4   -c 80 -t 2:20:00 -o ${outf}/mseg-lowres-test tool/train-qvga-mix-copy.sh  1080/mseg-lowres.yaml False exp ${WORK}/copies/test-new
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu4   -c 80 -t 2:20:00 -o ${outf}/mseg-lowres-1-test tool/train-qvga-mix-cd.sh  1080/mseg-lowres.yaml False exp ${WORK}/copies/test-new
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu4   -c 80 -t 2:20:00 -o ${outf}/mseg-lowres-2-test tool/train-qvga-mix-cd.sh  1080/mseg-lowres.yaml False exp ${WORK}/copies/test-new


sh tool/train-qvga-mix-copy.sh  1080/mseg-3m.yaml False exp ${WORK}/copies/test




