#!/bin/sh
cd "$(dirname "$0")/../models/BAM"

PARTITION=SaliencyPrediction

dataset=$1
exp_name=$2
net=$3 # vgg resnet50
GPU_ID=$4

arch=BAM

exp_dir=exp/${dataset}/${arch}/${exp_name}/${net} 
snapshot_dir=${exp_dir}/snapshot
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}_${net}.yaml
mkdir -p ${snapshot_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
# cp test.sh test.py ${config} ${exp_dir}

echo ${arch}
echo ${config}

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u test.py \
        --config=${config} \
        --arch=${arch} \
        2>&1 | tee ${result_dir}/test-$now.log