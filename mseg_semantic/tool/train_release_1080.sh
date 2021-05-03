export outf=0424_release
mkdir ${outf}

# all is so-called "lowres", 13801-13808

# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu18   -c 80 -t 2-00:00:00 -o ${outf}/mseg-lowres-3m tool/train-qvga-mix-copy.sh  1080_release/mseg-lowres-3m.yaml False exp ${WORK}/copies/final_train/1080_release/mseg-lowres-3m
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu18   -c 80 -t 2-00:00:00 -o ${outf}/mseg-lowres-3m-1 tool/train-qvga-mix-cd.sh  1080_release/mseg-lowres-3m.yaml False exp ${WORK}/copies/final_train/1080_release/mseg-lowres-3m
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu18   -c 80 -t 2-00:00:00 -o ${outf}/mseg-lowres-3m-2 tool/train-qvga-mix-cd.sh  1080_release/mseg-lowres-3m.yaml False exp ${WORK}/copies/final_train/1080_release/mseg-lowres-3m
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu18   -c 80 -t 2-00:00:00 -o ${outf}/mseg-lowres-3m-3 tool/train-qvga-mix-cd.sh  1080_release/mseg-lowres-3m.yaml False exp ${WORK}/copies/final_train/1080_release/mseg-lowres-3m
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu18   -c 80 -t 2-00:00:00 -o ${outf}/mseg-lowres-3m-4 tool/train-qvga-mix-cd.sh  1080_release/mseg-lowres-3m.yaml False exp ${WORK}/copies/final_train/1080_release/mseg-lowres-3m
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu18   -c 80 -t 2-00:00:00 -o ${outf}/mseg-lowres-3m-5 tool/train-qvga-mix-cd.sh  1080_release/mseg-lowres-3m.yaml False exp ${WORK}/copies/final_train/1080_release/mseg-lowres-3m



# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu8   -c 80 -t 2-00:00:00 -o ${outf}/mseg-unrelabeled tool/train-qvga-mix-copy.sh  1080_release/mseg-unrelabeled.yaml False exp ${WORK}/copies/final_train/1080_release/mseg-unrelabeled-1
# 14239
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu3   -c 80 -t 2-00:00:00 -o ${outf}/mseg-unrelabeled-1 tool/train-qvga-mix-cd.sh  1080_release/mseg-unrelabeled.yaml False exp ${WORK}/copies/final_train/1080_release/mseg-unrelabeled-1

# 14293-14297
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu19   -c 80 -t 2-00:00:00 -o ${outf}/mseg-720-3m tool/train-qvga-mix-copy.sh  720_release/mseg-3m.yaml False exp ${WORK}/copies/final_train/720_release/mseg-3m
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu19   -c 80 -t 2-00:00:00 -o ${outf}/mseg-720-3m-1 tool/train-qvga-mix-cd.sh  720_release/mseg-3m.yaml False exp ${WORK}/copies/final_train/720_release/mseg-3m
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu19   -c 80 -t 2-00:00:00 -o ${outf}/mseg-720-3m-2 tool/train-qvga-mix-cd.sh  720_release/mseg-3m.yaml False exp ${WORK}/copies/final_train/720_release/mseg-3m
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu19   -c 80 -t 2-00:00:00 -o ${outf}/mseg-720-3m-3 tool/train-qvga-mix-cd.sh  720_release/mseg-3m.yaml False exp ${WORK}/copies/final_train/720_release/mseg-3m
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu19   -c 80 -t 2-00:00:00 -o ${outf}/mseg-720-3m-4 tool/train-qvga-mix-cd.sh  720_release/mseg-3m.yaml False exp ${WORK}/copies/final_train/720_release/mseg-3m


# 14301-14304
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu4   -c 80 -t 2-00:00:00 -o ${outf}/mseg-480-3m tool/train-qvga-mix-copy.sh  480_release/mseg-3m.yaml False exp ${WORK}/copies/final_train/480_release/mseg-3m
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu4   -c 80 -t 2-00:00:00 -o ${outf}/mseg-480-3m-1 tool/train-qvga-mix-cd.sh  480_release/mseg-3m.yaml False exp ${WORK}/copies/final_train/480_release/mseg-3m
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu4   -c 80 -t 2-00:00:00 -o ${outf}/mseg-480-3m-2 tool/train-qvga-mix-cd.sh  480_release/mseg-3m.yaml False exp ${WORK}/copies/final_train/480_release/mseg-3m
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu4   -c 80 -t 2-00:00:00 -o ${outf}/mseg-480-3m-3 tool/train-qvga-mix-cd.sh  480_release/mseg-3m.yaml False exp ${WORK}/copies/final_train/480_release/mseg-3m



# 14308-14312
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu7   -c 80 -t 2-00:00:00 -o ${outf}/mseg-mgda tool/train-qvga-mix-copy.sh  1080_release/mseg-mgda.yaml True exp ${WORK}/copies/final_train/1080_release/mseg-mgda
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu7   -c 80 -t 2-00:00:00 -o ${outf}/mseg-mgda-1 tool/train-qvga-mix-cd.sh  1080_release/mseg-mgda.yaml True exp ${WORK}/copies/final_train/1080_release/mseg-mgda
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu7   -c 80 -t 2-00:00:00 -o ${outf}/mseg-mgda-2 tool/train-qvga-mix-cd.sh  1080_release/mseg-mgda.yaml True exp ${WORK}/copies/final_train/1080_release/mseg-mgda
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu7   -c 80 -t 2-00:00:00 -o ${outf}/mseg-mgda-3 tool/train-qvga-mix-cd.sh  1080_release/mseg-mgda.yaml True exp ${WORK}/copies/final_train/1080_release/mseg-mgda
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu7   -c 80 -t 2-00:00:00 -o ${outf}/mseg-mgda-4 tool/train-qvga-mix-cd.sh  1080_release/mseg-mgda.yaml True exp ${WORK}/copies/final_train/1080_release/mseg-mgda


# 14315-16
sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu24   -c 80 -t 2-00:00:00 -o ${outf}/mseg-baseline tool/train-qvga-mix-copy.sh  1080_release/mseg-baseline.yaml False exp ${WORK}/copies/final_train/1080_release/mseg-baseline
sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu24   -c 80 -t 2-00:00:00 -o ${outf}/mseg-baseline-1 tool/train-qvga-mix-cd.sh  1080_release/mseg-baseline.yaml False exp ${WORK}/copies/final_train/1080_release/mseg-baseline



# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu24   -c 80 -t 2-00:00:00 -o ${outf}/mseg-lowres tool/train-qvga-mix-copy.sh  1080_release/mseg-lowres.yaml False exp ${WORK}/copies/final_train/1080_release/mseg-lowres
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu24   -c 80 -t 2-00:00:00 -o ${outf}/mseg-lowres-1 tool/train-qvga-mix-cd.sh  1080_release/mseg-lowres.yaml False exp ${WORK}/copies/final_train/1080_release/mseg-lowres
# sbatch -p quadro --qos=normal --gres=gpu:8   -w isl-gpu24   -c 80 -t 2-00:00:00 -o ${outf}/mseg-lowres-2 tool/train-qvga-mix-cd.sh  1080_release/mseg-lowres.yaml False exp ${WORK}/copies/final_train/1080_release/mseg-lowres





