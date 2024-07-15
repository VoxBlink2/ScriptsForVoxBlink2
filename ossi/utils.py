import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def cosine(x,w):
    # x, w shape: [B, d], where B = batch size, d = feature dim.
    x_norm = F.normalize(x,dim=1)
    w_norm = F.normalize(w,dim=1)
    cos_sim = torch.mm(x_norm, w_norm.T).clamp(-1, 1)
    return cos_sim


def compute_dir_far(Gfeat, Glabel, Pfeat, Plabel,):
    num_cls = Plabel[-1].item()
    # num_cls = Plabel[-1]
    temp = torch.zeros(num_cls, Gfeat.size(1))
    for i in range(num_cls):
        mask = Glabel.eq(i)
        temp[i] = Gfeat[mask].mean(dim=0)  # make embd vector
    Gfeat = temp.clone()

    num_cls = Plabel[-1].item()
    # num_cls = Plabel[-1]
    Umask = Plabel.eq(num_cls)
    Klabel = Plabel[~Umask]
    Kfeat = Pfeat[~Umask]
    Ufeat = Pfeat[Umask]

    # compute cosine similarity
    Kcos = cosine(Kfeat, Gfeat)
    Ucos = cosine(Ufeat, Gfeat)

    # get prediction & confidence
    Kconf, Kidx = Kcos.max(1)
    Uconf, _ = Ucos.max(1)

    corr_mask = Kidx.eq(Klabel)
    dir_far_tensor = torch.zeros(1000, 3)  # intervals: 1000
    for i, th in enumerate(torch.linspace(Uconf.min(), Uconf.max(), 1000)):
        mask = (corr_mask) & (Kconf > th)
        dir = torch.sum(mask).item() / Kcos.size(0)
        far = torch.sum(Uconf > th).item() / Ucos.size(0)
        dir_far_tensor[i] = torch.FloatTensor([th, dir, far])  # [threshold, DIR, FAR] for each row
    return dir_far_tensor


def dir_at_far(dir_far_tensor,far):
    # deal with exceptions: there can be multiple thresholds that meets the given FAR (e.g., FAR=1.000)
    # if so, we must choose maximum DIR value among those cases
    abs_diff = torch.abs(dir_far_tensor[:,2]-far)
    minval = abs_diff.min()
    mask = abs_diff.eq(minval)
    dir_far = dir_far_tensor[mask]
    dir = dir_far[:,1].max().item()
    return dir


# area under DIR@FAR curve
def AUC(dir_far_tensor):
    auc = 0
    eps = 1e-5
    for i in range(dir_far_tensor.size(0)-1):
        if dir_far_tensor[i,1].ge(eps) and dir_far_tensor[i,2].ge(eps)\
                and dir_far_tensor[i+1,1].ge(eps) and dir_far_tensor[i+1,2].ge(eps):
            height = (dir_far_tensor[i,1] + dir_far_tensor[i+1,1])/2
            width = torch.abs(dir_far_tensor[i,2] - dir_far_tensor[i+1,2])
            auc += (height*width).item()
    return auc

def save_dir_far_curve(Gfeat, Glabel, Pfeat, Plabel, save_dir,save_name):
    cos_tensor = compute_dir_far(Gfeat, Glabel, Pfeat, Plabel, matcher='cos')
    cos_auc = AUC(cos_tensor)
    fig,ax = plt.subplots(1,1)
    ax.plot(cos_tensor[:,2], cos_tensor[:,1])
    ax.set_xscale('log')
    ax.set_xlabel('FAR')
    ax.set_ylabel('DIR')
    ax.legend(['cos-AUC: {:.3f}'.format(cos_auc)])
    ax.grid()
    fig.savefig(save_dir+'/'+save_name, bbox_inches='tight')

def save_dir_res(Gfeat, Glabel, Pfeat, Plabel, save_pic, save_res, fars):
    cos_tensor = compute_dir_far(Gfeat, Glabel, Pfeat, Plabel)
    cos_auc = AUC(cos_tensor)
    fig,ax = plt.subplots(1,1)
    ax.plot(cos_tensor[:,2], cos_tensor[:,1])
    ax.set_xscale('log')
    ax.set_xlabel('FAR')
    ax.set_ylabel('DIR')
    ax.legend(['cos-AUC: {:.3f}'.format(cos_auc)])
    ax.grid()
    fig.savefig(save_pic, bbox_inches='tight')
    with open(save_res,'a') as f:
        for far in fars:
            dir= dir_at_far(cos_tensor,far=far)
            f.write("\tFAR=%.4f:\tDIR=%.4f\n"%(far,dir))
            f.flush()
