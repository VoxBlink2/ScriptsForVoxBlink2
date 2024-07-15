import torch, torch.nn as nn, numpy as np,sys
from modules.front_resnet import ResNet34, ResNet100,ResNet293,block2module
import modules.pooling as pooling_func
import torch.nn.functional as F  
import profile 
####################################################################
##### ResNet-Based #######################################
####################################################################

class ResNet34_based(nn.Module):
    def __init__(self, in_planes, block_type, pooling_layer, embd_dim, acoustic_dim, featCal, dropout=0,**kwargs):
        super(ResNet34_based, self).__init__()
        print('ResNet34 based model with %s block and %s pooling' %(block_type, pooling_layer))
        self.featCal = featCal
        self.front = ResNet34(in_planes, block_type)
        self.pooling = getattr(pooling_func, pooling_layer)(in_planes,acoustic_dim)
        self.bottleneck = nn.Linear(self.pooling.out_dim, embd_dim)
        self.drop = nn.Dropout(dropout) if dropout else None
    def forward(self, x):
        x = self.featCal(x)
        x = self.front(x.unsqueeze(dim=1))
        x = self.pooling(x)
        if self.drop:
            x = self.drop(x)
        x = self.bottleneck(x)
        return x


class ResNet100_based(nn.Module):
    def __init__(self, in_planes, block_type, pooling_layer, embd_dim, acoustic_dim, featCal, dropout=0, **kwargs):
        super(ResNet100_based, self).__init__()
        print('ResNet100 based model with %s and %s ' %(block_type, pooling_layer))
        self.featCal = featCal
        self.front = ResNet100(in_planes, block_type)
        block_expansion = block2module[block_type].expansion
        self.pooling = getattr(pooling_func, pooling_layer)(in_planes*block_expansion,acoustic_dim)
        self.bottleneck = nn.Linear(self.pooling.out_dim, embd_dim)
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x):
        x = self.featCal(x)
        x = self.front(x.unsqueeze(dim=1))
        x = self.pooling(x)
        if self.drop:
            x = self.drop(x)
        x = self.bottleneck(x)
        
        return x

class ResNet293_based(nn.Module):
    def __init__(self, in_planes, block_type, pooling_layer, embd_dim, acoustic_dim, featCal, dropout=0,**kwargs):
        super(ResNet293_based, self).__init__()
        print('ResNet293 based model with %s and %s ' %(block_type, pooling_layer))
        self.featCal = featCal
        self.front = ResNet293(in_planes, block_type)
        block_expansion = block2module[block_type].expansion
        self.pooling = getattr(pooling_func, pooling_layer)(in_planes*block_expansion,acoustic_dim)
        self.bottleneck = nn.Linear(self.pooling.out_dim, embd_dim)
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x):
        x = self.featCal(x)
        x = self.front(x.unsqueeze(dim=1))
        x = self.pooling(x)
        if self.drop:
            x = self.drop(x)
        x = self.bottleneck(x)
        return x

####################################################################
##### ECAPA_TDNN #######################################
####################################################################

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
 
####################################################################
##### ECAPA_TDNN #######################################
####################################################################

import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 


class ECAPA_TDNN(nn.Module):

    def __init__(self, C, featCal):

        super(ECAPA_TDNN, self).__init__()
        self.featCal = featCal
        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)


    def forward(self, x):
        x = self.featCal(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x
    
if __name__ == '__main__':
    model = ResNet34_based(64,'base',pooling_layer='GSP',embd_dim=256,acoustic_dim=80)
    input = torch.rand([1, 16000])
    flops, params = profile(model, inputs=(input, ))
    print('FLOPs = ' + str(flops/1024**3) + 'G')
    print('Params = ' + str(params/1024**2) + 'M')
