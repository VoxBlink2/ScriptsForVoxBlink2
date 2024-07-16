#! /usr/bin/env python3
import argparse, numpy as np,os
import torch, torch.nn as nn
from spk_veri_metric import SVevaluation
from torch.utils.data import DataLoader
from dataset import WavDataset
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml

parser = argparse.ArgumentParser(description="ASV evaluation")
parser.add_argument('--yaml_path',required=True,  type=str)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--num_workers', default=10, type=int)
parser.add_argument('--output_path', required=False, type=str)

args = parser.parse_args()
# feature
with open(args.yaml_path, 'r') as f:
    yaml_strings = f.read()
    hparams = load_hyperpyyaml(yaml_strings)


val_utt = [line.split()[0] for line in open('%s/wav.scp' % hparams['val_name'])]

# dataset
val_dataset = WavDataset(
    [
        line.split()
        for line in open(
           '%s/wav.scp' % hparams['val_name']
        )
    ],
    norm_type=hparams['norm_type']
)
val_dataloader = DataLoader(
    val_dataset,
    num_workers=args.num_workers,
    shuffle=False,
    batch_size=1,
)

model = hparams['model']
state_dict = torch.load(hparams['ckpt_path'],map_location=args.device)
model.load_state_dict({k.replace('module.',''):v for k,v in state_dict.items()},strict=False)

model = model.to(args.device)

model.eval()
embd_stack = np.zeros([0, hparams['embd_dim']])
embd_dict = {}
with torch.no_grad():
    for j, (feat, utt) in enumerate(tqdm(val_dataloader)):
        embd = model(feat.to(args.device)).cpu().numpy()
        embd_dict[utt[0]] = embd
        embd_stack = np.concatenate((embd_stack,embd))
if os.path.exists('%s/trials' % hparams['val_name']):
    eer_cal = SVevaluation('%s/trials' % hparams['val_name'], val_utt,ptar=[0.01])
    eer_cal.update_embd(embd_stack)
    eer, cost = eer_cal.eer_cost()
if args.output_path:
    np.save(os.path.join(args.output_path,'embd.npy'),embd_dict,allow_pickle=True)
print(eer,cost)
