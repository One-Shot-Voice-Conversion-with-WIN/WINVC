#!/bin/bash

#source activate pgan:15234上单独安装了环境
PYTHON=/home/hsj/anaconda3/envs/pgan/bin/python

root=/home/hsj/2021/one_shot/VCTK/task15


scp_dir=$root/feats.scp
mel_files=$root/converted/
output_dir=$root/wav


checkpoint_=/home/hsj/2021/one_shot/convert_decode/pgan_decode/pgan/vctk1000ckpt/checkpoint-1000000steps.pkl
config=/home/hsj/2021/one_shot/convert_decode/pgan_decode/pgan/vctk1000ckpt/config.yml


main_script=/home/hsj/2021/one_shot/convert_decode/pgan_decode/pgan_decode.py


echo "执行的文件名：$0";
echo "第一个参数为：$1";
echo "第二个参数为：$2";
echo "第三个参数为：$3";

CUDA_VISIBLE_DEVICES=$3 $PYTHON $main_script \
                    --feats-scp $1 \
                    --outdir $2 \
                    --checkpoint $checkpoint_ \
                    --config $config \
                    --verbose 1
                   