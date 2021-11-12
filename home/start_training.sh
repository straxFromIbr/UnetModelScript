#! /bin/bash
set -euxo pipefail
cd "$(dirname "$0")"

tensorboard --bind_all --logdir ./logs &
python3 ./train.py
pkill tensorboard
tensorboard dev upload --logdir ./logs --one_shot

