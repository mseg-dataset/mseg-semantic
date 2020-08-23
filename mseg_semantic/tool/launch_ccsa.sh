export outf=1122
mkdir ${outf}

# v1 uses 1000 pairs
# sbatch -p quadro --gres=gpu:6 -c 60 -t 2-00:00:00 -o ${outf}/three-1.6-ccsa tool/train-ccsa-qvga-mix.sh three-1.6-ccsa.yaml False exp-ccsa-v1 ${WORK}/supp/three-1.6-ccsa
# may have gotten polluted

# V2 uses 100 pairs

# v4 has 1,000 pairs for sure

# v5 has 10,000 pairs for sure

# v6 has 1,000 pairs with alpha 0.5

# v7 has 1,000 pairs with alpha 0.1

# v8 alpha = 0 with 1000 pairs, should be no DG effectively

sbatch -p quadro --gres=gpu:6 -c 60 -t 2-00:00:00 -o ${outf}/three-1.6-ccsa tool/train-ccsa-qvga-mix.sh three-1.6-ccsa.yaml False exp-ccsa-v9 ${WORK}/supp/three-1.6-ccsa-v9
