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

class SE_Res2Block(nn.Module):
    
    def __init__(self,k=3,d=2,s=8,channel=512,bottleneck=128):
        super(SE_Res2Block,self).__init__()
        self.k = k
        self.d = d
        self.s = s
        if self.s == 1:
            self.nums = 1
        else:
            self.nums = self.s - 1
            
        self.channel = channel
        self.bottleneck = bottleneck
        
        self.conv1 = nn.Conv1d(self.channel,self.channel,kernel_size=1,dilation=1)
        self.bn1 = nn.BatchNorm1d(self.channel)
        
        self.convs = []
        self.bns = []
        for i in range(self.s):
            self.convs.append(nn.Conv1d(int(self.channel/self.s), int(self.channel/self.s), kernel_size=self.k, dilation=self.d, padding=self.d, bias=False,padding_mode='reflect'))
            self.bns.append(nn.BatchNorm1d(int(self.channel/self.s)))
            
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)
        
        self.conv3 = nn.Conv1d(self.channel,self.channel,kernel_size=1,dilation=1)
        self.bn3 = nn.BatchNorm1d(self.channel)
        
        self.fc1 = nn.Linear(self.channel,self.bottleneck,bias=True)
        self.fc2 = nn.Linear(self.bottleneck,self.channel,bias=True)
        
    def forward(self,x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))

        spx = torch.split(out, int(self.channel/self.s), 1)
        for i in range(1,self.nums+1):
            if i==1:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = F.relu(self.bns[i](sp))
            if i==1:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        if self.s != 1 :
            out = torch.cat((out, spx[0]),1)
        
        out = F.relu(self.bn3(self.conv3(out)))
        out_mean = torch.mean(out,dim=2)
        s_v = torch.sigmoid(self.fc2(F.relu(self.fc1(out_mean))))
        out = out * s_v.unsqueeze(-1)
        out += residual
        #out = F.relu(out)
        return out


class Classic_Attention(nn.Module):
    def __init__(self,input_dim, embed_dim, attn_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.lin_proj = nn.Linear(input_dim,embed_dim)
        self.v = torch.nn.Parameter(torch.randn(embed_dim))
    
    def forward(self,inputs):
        lin_out = self.lin_proj(inputs)
        v_view = self.v.unsqueeze(0).expand(lin_out.size(0), len(self.v)).unsqueeze(2)
        attention_weights = torch.tanh(lin_out.bmm(v_view).squeeze(-1))
        attention_weights_normalized = F.softmax(attention_weights,1)
        #attention_weights_normalized = F.softmax(attention_weights)
        return attention_weights_normalized

class Attentive_Statictics_Pooling(nn.Module):
    
    def __init__(self,channel=1536,R_dim_self_att=128):
        super(Attentive_Statictics_Pooling,self).__init__()
        
        self.attention = Classic_Attention(channel,R_dim_self_att)
    
    def weighted_sd(self,inputs,attention_weights, mean):
        el_mat_prod = torch.mul(inputs,attention_weights.unsqueeze(2).expand(-1,-1,inputs.shape[-1]))
        hadmard_prod = torch.mul(inputs,el_mat_prod)
        variance = torch.sum(hadmard_prod,1) - torch.mul(mean,mean)
        return variance    
    
    def stat_attn_pool(self,inputs,attention_weights):
        el_mat_prod = torch.mul(inputs,attention_weights.unsqueeze(2).expand(-1,-1,inputs.shape[-1]))
        mean = torch.mean(el_mat_prod,1)
        variance = self.weighted_sd(inputs,attention_weights,mean)
        stat_pooling = torch.cat((mean,variance),1)
        return stat_pooling
    
    def forward(self,x):
        attn_weights = self.attention(x)
        stat_pool_out = self.stat_attn_pool(x,attn_weights)
        
        return stat_pool_out
    
class ECAPA_TDNN(nn.Module):
    
    def __init__(self,in_dim,hidden_dim,scale,bottleneck,embedding_size,featCal):
        
        super(ECAPA_TDNN,self).__init__()
        self.featCal = featCal
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.scale = scale
        self.bottleneck = bottleneck
        self.embedding_size = embedding_size
        
        self.conv1 = nn.Conv1d(in_dim,hidden_dim,kernel_size=5,dilation=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.block1 = SE_Res2Block(k=3,d=2,s=self.scale,channel=self.hidden_dim,bottleneck=self.bottleneck)
        self.block2 = SE_Res2Block(k=3,d=3,s=self.scale,channel=self.hidden_dim,bottleneck=self.bottleneck)
        self.block3 = SE_Res2Block(k=3,d=4,s=self.scale,channel=self.hidden_dim,bottleneck=self.bottleneck)
        self.conv2 = nn.Conv1d(self.hidden_dim*3,self.hidden_dim*3,kernel_size=1,dilation=1)
        
        self.ASP = Attentive_Statictics_Pooling(channel=self.hidden_dim*3,R_dim_self_att=self.bottleneck)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim*3*2)
        
        self.fc = nn.Linear(self.hidden_dim*3*2,self.embedding_size)
        self.bn3 = nn.BatchNorm1d(self.embedding_size)
        
    def forward(self,x):
        x = self.featCal(x)
        x = x.transpose(1,2)
        y = F.relu(self.bn1(self.conv1(x)))
        y_1 = self.block1(y)
        y_2 = self.block2(y_1)
        y_3 = self.block3(y_2)
        out = torch.cat((y_1, y_2,y_3), 1)
        out = F.relu(self.conv2(out))
        out = self.bn2(self.ASP(out.transpose(1,2)))
        out = self.bn3(self.fc(out))
        return out

if __name__ == '__main__':
    model = ResNet34_based(64,'base',pooling_layer='GSP',embd_dim=256,acoustic_dim=80)
    input = torch.rand([1, 16000])
    flops, params = profile(model, inputs=(input, ))
    print('FLOPs = ' + str(flops/1024**3) + 'G')
    print('Params = ' + str(params/1024**2) + 'M')
