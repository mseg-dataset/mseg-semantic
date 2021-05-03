#!/bin/sh
export outf=0711-2
sbatch -C turing -p gpu --gres=gpu:8 -c 80 -o ${outf}/city_18 tool/train.sh cityscapes_18 pspnet50
sbatch -C turing -p gpu --gres=gpu:8 -c 80 -o ${outf}/nyu_36 tool/train.sh nyudepthv2_36 pspnet50

#7846 and 7847

# sbatch -C turing -p gpu --gres=gpu:8 -c 80 -o ${outf}/map-coco      tool/train_flatmix.sh mix flat-map-coco
# sbatch           -p gpu --gres=gpu:8 -c 80 -o ${outf}/coco-scan     tool/train_flatmix.sh mix flat-coco-scan
# sbatch           -p gpu --gres=gpu:8 -c 80 -o ${outf}/map-scan      tool/train_flatmix.sh mix flat-map-scan
# sbatch           -p gpu --gres=gpu:8 -c 80 -o ${outf}/map           tool/train_flatmix.sh mix flat-map
# sbatch           -p gpu --gres=gpu:8 -c 80 -o ${outf}/coco          tool/train_flatmix.sh mix flat-coco
# sbatch           -p gpu --gres=gpu:8 -c 80 -o ${outf}/scan          tool/train_flatmix.sh mix flat-scan

# 2-8 above

# sbatch -p quadro --gres=gpu:8 -c 80 -o ${outf}/coco-scan-2     tool/train_flatmix.sh mix flat-coco-scan # 7786
# sbatch -C turing -p gpu --gres=gpu:8 -c 80 -o ${outf}/map-scan-2      tool/train_flatmix.sh mix flat-map-scan # 7781
