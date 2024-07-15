#! /usr/bin/env python3
import argparse, numpy as np
import torch, torch.nn as nn
from spk_veri_metric import SVevaluation
from torch.utils.data import DataLoader
from dataset import WavDataset
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml

parser = argparse.ArgumentParser(description="Speaker Embedding Extraction")
parser.add_argument('--yaml_path', default='asv/conf/resnet34.yaml', type=str)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--num_workers', default=0, type=int)

args = parser.parse_args()
# feature
with open(args.yaml_path, 'r') as f:
    yaml_strings = f.read()
    hparams = load_hyperpyyaml(yaml_strings)


val_utt = [line.split()[0] for line in open('data/%s/wav.scp' % hparams['val_name'])]
eer_cal = SVevaluation('data/%s/trials' % hparams['val_name'], val_utt,ptar=[0.01, 0.001])

# dataset
val_dataset = WavDataset(
    [
        line.split()
        for line in open(
           'data/%s/wav.scp' % hparams['val_name']
        )
    ],
)
val_dataloader = DataLoader(
    val_dataset,
    num_workers=args.num_workers,
    shuffle=False,
    batch_size=1,
)

model = hparams['model']
model.load_state_dict(
    torch.load(
        hparams['ckpt_path'],map_location=args.device
    ),strict=False
)

model = model.to(args.device)

model.eval()
embd_stack = np.zeros([0, hparams['embd_dim']])
embd_dict = {}
with torch.no_grad():
    for j, (feat, utt) in enumerate(tqdm(val_dataloader)):
        embd = model(feat.to(args.device)).cpu().numpy()
        embd_stack = np.concatenate((embd_stack,embd))
eer_cal.update_embd(embd_stack)
eer, cost = eer_cal.eer_cost()
print(eer,cost)


