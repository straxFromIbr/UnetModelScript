#! /bin/bash
set -euxo pipefail
cd "$(dirname "$0")"

python3 ./train.py \
        --datadir '/mass_roads/train' --suffix 'png' --dsrate=0.5 \
        --logdir 'DAF-Focal-E20' \
        --loss 'Focal' \
        --epochs 20

#python3 ./train.py \
#        --datadir '/mass_roads/train' --suffix 'png' --dsrate=0.5 \
#        --logdir 'DAF-BCEDICE-E20' \
#        --loss 'BCEwithDICE' \
#        --epochs 20
#
#
#python3 ./train.py \
#        --datadir '/mass_roads/train' --suffix 'png' --dsrate=0.5 \
#        --logdir 'DAF-Tversky0.7-E20' \
#        --loss 'Tversky' \
#        --epochs 20

#python3 ./train.py \
#        --datadir '/mass_roads/train' --suffix 'png' \
#        --logdir 'cmTn4-d-E30_MR' \
#        --use_dice \
#        --use_cutmix \
#        --nbmix 4 \
#        --epochs 30

#python3 ./train.py \
#        --datadir '/mass_roads/train' --suffix 'png' \
#        --logdir 'cmF-d-E30_MR' \
#        --use_dice \
#        --epochs 30

#python3 ./train.py \
#    --datadir '/dataset/train' \
#    --logdir 'testing_ft' \
#    --use_dice \
#    --use_cutmix \
#    --epochs 10

#sleep 60
#python3 ./train.py --logdir 'cmTn3-a7-E30_MR' --use_cutmix --alpha 0.7 --epochs 30
#sleep 60
#python3 ./train.py --logdir 'cmTn3-a9-E30_MR' --use_cutmix --alpha 0.9 --epochs 30
#sleep 60

