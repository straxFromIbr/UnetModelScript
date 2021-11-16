#! /bin/bash
set -euxo pipefail
cd "$(dirname "$0")"

tensorboard --bind_all --logdir ./logs &
python3 ./train.py TVERSKY
python3 ./train.py DICE
python3 ./train.py FOCAL
pkill tensorboard
tensorboard dev upload --logdir ./logs --one_shot

