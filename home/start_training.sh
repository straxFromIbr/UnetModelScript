#! /bin/bash
set -euxo pipefail
cd "$(dirname "$0")"

tensorboard dev upload --logdir ./logs --name '実験' --description 'CutMixはOFFに、TverskyのAlpha値を変えて試してみる' &
python3 ./train.py --logdir 'cmF-a3-E20' --alpha 0.3 --epochs 50
python3 ./train.py --logdir 'cmF-a5-E20' --alpha 0.5 --epochs 50
python3 ./train.py --logdir 'cmF-a7-E20' --alpha 0.7 --epochs 50
pkill tensorboard
#tensorboard dev upload --logdir ./logs --one_shot

