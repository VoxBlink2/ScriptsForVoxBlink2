import torch, torch.nn as nn, torch.nn.functional as F

class GSP(nn.Module):
    # GlobalStatsPool
    def __init__(self,):
        super(GSP, self).__init__()

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        out = torch.cat([x.mean(dim=2), x.std(dim=2)], dim=1)
        return out

class ASP(nn.Module):
    # Attentive statistics pooling
    def __init__(self,in_planes,acoustic_dim):
        super(ASP, self).__init__()
        outmap_size = int(acoustic_dim/8)
        self.out_dim = in_planes*8 * outmap_size * 2
        
        self.attention = nn.Sequential(
                        nn.Conv1d(in_planes*8 * outmap_size, 128, kernel_size=1),
                        nn.ReLU(),
                        nn.BatchNorm1d(128),
                        nn.Conv1d(128, in_planes*8 * outmap_size, kernel_size=1),
                        nn.Softmax(dim=2),
                        )
        
    def forward(self, x):
        x = x.reshape(x.size()[0],-1,x.size()[-1])
        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-5) )
        x = torch.cat((mu,sg),1)
        
        x = x.view(x.size()[0], -1)
        return x
   