#! /bin/bash
set -euxoC pipefail
cd "$(dirname "$0")"

dataset='ibaraki'
model_root='../../experiments_autolr/checkpoints'
csv_root='../../モデル実験/'
python='/Users/hagayuya/mambaforge/envs/tensorflow/bin/python'

# model_name='DICE-DA_GEO-IM_1.0-E100_base1.0'
# $python explore_model.py \
#   --dataset "${dataset}" \
#   --model_root "${model_root}" \
#   --model_name "${model_name}" \
#   --csv_path "${csv_root}/${model_name}.csv"

# model_name='DICE-DA_GEO-IM_1.0-E100_base0.5'
# $python explore_model.py \
#   --dataset "${dataset}" \
#   --model_root "${model_root}" \
#   --model_name "${model_name}" \
#   --csv_path "${csv_root}/${model_name}.csv"

# model_name='DICE-DA_GEO-IM_1.0-E100_base_nopret'
# $python explore_model.py \
#   --dataset "${dataset}" \
#   --model_root "${model_root}" \
#   --model_name "${model_name}" \
#   --csv_path "${csv_root}/${model_name}.csv"

# model_name='DICE-DA_GEO-IM_0.5-E100_base1.0'
# $python explore_model.py \
#   --dataset "${dataset}" \
#   --model_root "${model_root}" \
#   --model_name "${model_name}" \
#   --csv_path "${csv_root}/${model_name}.csv"

# model_name='DICE-DA_GEO-IM_0.5-E100_base0.5'
# $python explore_model.py \
#   --dataset "${dataset}" \
#   --model_root "${model_root}" \
#   --model_name "${model_name}" \
#   --csv_path "${csv_root}/${model_name}.csv"

# model_name='DICE-DA_GEO-IM_1.0-E100_base_nopret'
# $python explore_model.py \
#   --dataset "${dataset}" \
#   --model_root "${model_root}" \
#   --model_name "${model_name}" \
#   --csv_path "${csv_root}/${model_name}.csv"

model_root='../../experiments_bases/checkpoints'
model_name='DICE-DA_GEO_CH-MR9_0.5-E50'
dataset="mass_roads"
$python explore_model.py \
  --epochs=50 \
  --dataset="${dataset}" \
  --model_root="${model_root}" \
  --model_name="${model_name}" \
  --csv_path="${csv_root}/${model_name}.csv"

model_name='DICE-DA_GEO_CH-MR9_1.0-E50'
$python explore_model.py \
  --epochs=50 \
  --dataset="${dataset}" \
  --model_root="${model_root}" \
  --model_name="${model_name}" \
  --csv_path="${csv_root}/${model_name}.csv"
