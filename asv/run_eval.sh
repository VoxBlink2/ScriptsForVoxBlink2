#!/bin/bash
if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
    device="cuda"
else
    device="cpu"
fi
num_workers=4
yaml_conf=conf/resnet34.yaml
python ./eval_ckpt.py --num_workers=${num_workers} --device=${device} --yaml_path=${yaml_conf}