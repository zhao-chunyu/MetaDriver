#!/bin/sh
cd "$(dirname "$0")/../models/HDMNet"

PARTITION=SaliencyPrediction

dataset=$1 # metadada
exp_name=$2 # split0
net=$3 # vgg resnet50
GPU_ID=$4 # 0/1/2/... single gpu


arch=HDMNet
exp_dir=exp/${dataset}/${exp_name}/${net}/
snapshot_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}_${net}_manet.yaml
mkdir -p ${snapshot_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
#cp train.sh train.py ${config} ${exp_dir}
#cp -r model ${exp_dir}/src

echo ${arch}
echo ${config}


CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u test.py \
        --config=${config} \
        --arch=${arch} \
        2>&1 | tee ${result_dir}/train-$now.log
