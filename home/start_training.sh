#! /bin/bash
set -euxo pipefail
cd "$(dirname "$0")"

date='211221'
pret='DA_CM3RFZ-E100-MR-base'
epoch='100'
python3 ./train.py \
        --datadir '/tokai_katsuta_ds_s17/train' --suffix 'jpg' \
        --logdir 'DA_RF-E100-tokai17-pret_base' \
        --pretrained \
		    "/results/${date}/checkpoints/${pret}/${pret}_${epoch}" \
        --loss 'DICE' \
        --epochs 100 \
        --augment --rotate --flip #\
                  #--use_cutmix --nbmix 4

        #--pretrained \
		#"/results/${date}/checkpoints/${pret}/${pret}_${epoch}" \
        #--pretrained '/results/211220/checkpoints/DA_CM3RF-DICE-E100_base/DA_CM3RF-DICE-E100_base_050' \
        # --pretrained '/results/211220/checkpoints/DA_CM3RF-DICE-E100_base/DA_CM3RF-DICE-E100_base_050' \
#python3 ./train.py \
#        --datadir '/mass_roads/train' --suffix 'png' \
#        --logdir 'DA_CM3RF-DICE-E100_base' \
#        --loss 'DICE' \
#        --epochs 2\
#        --augment --rotate --flip \
#                  --use_cutmix --nbmix 4

