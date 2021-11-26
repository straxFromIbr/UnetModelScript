#! /bin/bash
set -euxo pipefail
cd "$(dirname "$0")"

python3 ./train.py --logdir 'cmTn3-a7-E20_MR' --use_cutmix --alpha 0.7 --epochs 20
#python3 ./train.py --logdir 'cmF-a5-E20' --alpha 0.5 --epochs 50
#python3 ./train.py --logdir 'cmF-a7-E20' --alpha 0.7 --epochs 50

