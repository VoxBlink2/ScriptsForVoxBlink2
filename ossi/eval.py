import numpy as np
import torch,os,argparse
from tqdm import tqdm
from ossi.utils import save_dir_res
parser = argparse.ArgumentParser(description='Process prompt search agent arguments')
parser.add_argument('--data_path', 
                    type=str, 
                    default='data/ossi',
                    help='data_path')  
parser.add_argument('--embd_path', 
                    type=str, 
                    help='embd_path')  
parser.add_argument('--eval_mode', 
                    type=str, 
                    help='[small,medium,large]')  
parser.add_argument('--spk_mode', 
                    type=str, 
                    help='[perspk1,perspk3,perspk5]')  
parser.add_argument('--output_path', 
                    type=str, 
                    help='path to output results')  
                    
if __name__ == '__main__':
    args = parser.parse_args()
    embd_dict = np.load(args.embd_path,allow_pickle=True).item()
    gallery = os.path.join(args.data_path,args.eval_mode,args.spk_mode,'gallery')
    probe = os.path.join(args.data_path,args.eval_mode,args.spk_mode,'probe')
    Gfeat = torch.FloatTensor([])
    Glabel = torch.LongTensor([])
    for line in open(gallery):
        [utt,label] = line.strip().split()
        feat = torch.from_numpy(embd_dict[utt])
        label = torch.tensor([int(label)])
        Gfeat = torch.cat((Gfeat, feat), dim=0)
        Glabel = torch.cat((Glabel, label), dim=0)
    Pfeat = torch.FloatTensor([])
    Plabel = torch.LongTensor([])
    for line in tqdm(open(probe)):
        [utt,label] = line.strip().split()
        feat = torch.from_numpy(embd_dict[utt])
        label = torch.tensor([int(label)])
        Pfeat = torch.cat((Pfeat, feat), dim=0)
        Plabel = torch.cat((Plabel, label), dim=0)
    save_dir_res(Gfeat, Glabel, Pfeat, Plabel, 
                 os.path.join(args.output_path,'res_fig.jpg'),
                 os.path.join(args.output_path,'res.txt'),
                 [0.1,0.01,0.001]
                 )
    
