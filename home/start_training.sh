#! /bin/bash
set -euxo pipefail
cd "$(dirname "$0")"

python3 ./train.py \
        --datadir '/mass_roads/train' \
        --logdir 'testing_pt' \
        --use_dice \
        --use_cutmix \
        --epochs 10

python3 ./train.py \
    --datadir '/dataset/train' \
    --logdir 'testing_ft' \
    --use_dice \
    --use_cutmix \
    --epochs 10

#sleep 60
#python3 ./train.py --logdir 'cmTn3-a7-E30_MR' --use_cutmix --alpha 0.7 --epochs 30
#sleep 60
#python3 ./train.py --logdir 'cmTn3-a9-E30_MR' --use_cutmix --alpha 0.9 --epochs 30
#sleep 60

