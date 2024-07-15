#!/bin/bash

data_path=data/ossi
embd_path=ossi/example.npy
eval_mode=small
spk_mode=perspk3
output_path=.

python ossi/eval.py --data_path=${data_path} \
    --embd_path=${embd_path} --eval_mode=${eval_mode} \
    --spk_mode=${spk_mode} --output_path=$output_path
