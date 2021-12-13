#! /bin/bash
set -euxo pipefail
cd "$(dirname "$0")"

python3 ./train.py \
        --datadir '/mass_roads/train' --suffix 'png' \
        --logdir 'DA_F-DICE-E50' \
        --loss 'DICE' \
        --epochs 50 \
        --augment --flip
                
#python3 ./train.py \
#        --datadir '/mass_roads/train' --suffix 'png' \
#        --logdir 'NDA-DICE-E50' \
#        --loss 'DICE' \
#        --epochs 50
#
#python3 ./train.py \
#        --datadir '/mass_roads/train' --suffix 'png' \
#        --logdir 'NDA-DICE-E50-EQ' \
#        --loss 'DICE' \
#        --epochs 50 \
#        --eq
