import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import numpy as np
from torch.nn import functional as F
import time
from model.gcn import GCN
        
n_head = 12
n_layer = 2
dropout = 0.3
gcn_stages = 8


def afnc():
    return nn.Sigmoid()


def init_2(module):

    if hasattr(module, "weight"):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    if hasattr(module, "bias") and module.bias is not None:
        torch.nn.init.zeros_(module.bias)


class TCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TCN, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels*5, 10, 1, 10)
        self.conv_out = nn.Conv1d(out_channels*5, out_channels, 20, 1, 4)
        
        self.act = afnc()
        
        init_2(self.conv)
        init_2(self.conv_out)
        
    def forward(self, x):
        x = self.conv(x)

        x = self.act(x)
        
        x = self.conv_out(x)
        
        return x


class Head(nn.Module):

    def __init__(self, head_size, block_size, n_embd):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
            
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, block_size, n_embd) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        init_2(self.proj)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class TCNModule(nn.Module):

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            TCN(n_embd, 2 * n_embd),
            afnc(),
            TCN(n_embd * 2, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class FCModule(nn.Module):

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 2 * n_embd),
            afnc(),
            nn.Linear(n_embd * 2, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TCNBlock(nn.Module):

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = TCNModule(block_size)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class FCBlock(nn.Module):

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = FCModule(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)        
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

    
class ProjectionHead(nn.Module):

    def __init__(self, in_embd_size, out_embd_size, in_block_size, out_block_size):
        super().__init__()

        self.s_fc = nn.Linear(in_embd_size, out_embd_size)

        self.t_fc = nn.Sequential(
            nn.Linear(in_block_size, out_block_size*4),
            nn.Linear(out_block_size*4, out_block_size)
        )        
        
    def forward(self, x):
        
        x = self.s_fc(x)
        x = self.t_fc(x.transpose(1, 2)).transpose(1, 2)
        
        return x

    
class SceneModule(nn.Module):

    def __init__(self, config, in_size=16, out_size=14, n_in_embd=78, n_out_embd=78):
        super().__init__()
        
        self.config = config

        self.ph_in = ProjectionHead(n_in_embd, n_out_embd, in_size, out_size)

        self.hot = nn.Linear(config.N, config.J*3)
        self.t_hot = nn.Linear(1, out_size)
        
        self.gcn = GCN(input_feature=out_size, hidden_feature=out_size*2, p_dropout=dropout, num_stage=gcn_stages, node_n=n_out_embd)

    def forward(self, sequences):
                
        joined = torch.cat(sequences, dim=-1)
        
        joined = self.ph_in(joined)

        joined = self.gcn( joined.transpose(1, 2) ).transpose(1, 2)
        
        one_hots = []
        for n in range(self.config.N):
            n_hot = torch.zeros(joined.shape[0]).to(torch.int64) + n
            n_hot = self.hot(F.one_hot(n_hot, num_classes=self.config.N).float().to(self.config.device))            
            n_hot = self.t_hot(n_hot.unsqueeze(-2).transpose(1, 2)).transpose(1, 2)
            one_hots.append(n_hot)

        return joined, one_hots
    
    
class MTTPF(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        pose_size = config.J * 3
        
        n_embd = pose_size * 2 * 2
        scene_embd = pose_size * config.N * 2
        self.T = config.input_len + config.output_len

        self.ph_in = ProjectionHead(pose_size*2, n_embd, self.T, self.T)
                
        EMBD = n_embd + scene_embd + config.J*3
        
        if "attention" not in self.config.ablation_exclude:
            self.position_embedding_table = nn.Embedding(self.T, EMBD) # (T, C)

            self.blocks = nn.Sequential(*[TCNBlock(EMBD, n_head, self.T) if i%2==0 else FCBlock(EMBD, n_head, self.T)  for i in range(n_layer)])
            
        self.scene = SceneModule(config, in_size=self.T, out_size=self.T, n_in_embd=config.N*(config.J*3)*2, n_out_embd=scene_embd)
        
        if "temporal_gcn" not in self.config.ablation_exclude:
            self.gcn = GCN(input_feature=EMBD, hidden_feature=EMBD, p_dropout=dropout, num_stage=gcn_stages, node_n=self.T)
        
        self.fc_final = nn.Linear(EMBD*2, EMBD)

        self.ph_out = ProjectionHead(EMBD, pose_size, self.T, self.T)
        
        self.scene_j_distances = nn.Sequential(nn.Linear(scene_embd, scene_embd),
                                               afnc(),
                                               nn.Linear(scene_embd, self.config.J*3)
                                              )

    def w_vel(self, x):
        vel = x.clone()
        vel = vel[:, 1:] - vel[:, :-1]
        vel = torch.cat((vel[:, 0:1], vel), dim=1)
        
        idx = torch.cat((x, vel), dim=-1)
        return idx
            
    def forward(self, sequences):
        scene, one_hots = self.scene([self.w_vel(x) for x in sequences])
        
        return [self.fwd(x, scene, x_hot) for x, x_hot in zip(sequences, one_hots)], self.scene_j_distances(scene)
    
    def fwd(self, x, scene, x_hot):
        
        x = self.w_vel(x)
                
        x_emb = self.ph_in(x)
        
        if "temporal_gcn" not in self.config.ablation_exclude:
            gcn_out = self.gcn( torch.cat((x_emb, scene, x_hot), dim=-1) )
        else:
            gcn_out = torch.cat((x_emb, scene, x_hot), dim=-1)
        
        if "attention" not in self.config.ablation_exclude:
            pos_emb = self.position_embedding_table(torch.arange(self.T, device=self.config.device)) # (T,C)
            x = torch.cat((x_emb, scene, x_hot), dim=-1) + pos_emb # (B,T,C)
            x = self.blocks(x) # (B,T,C)
            x = self.fc_final(torch.cat((gcn_out, x), axis=-1))
            
        else:
            x = gcn_out
        
        xout = self.ph_out(x)
                                   
        xout = xout.reshape(-1, self.T, self.config.J, 3)     

        return xout


class MTTPFModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config

        self.m = MTTPF(config).float()
        
    def padd(self, x):

        x = x.reshape(-1, self.config.input_len, self.config.J, 3)

        newx = x[:, -1:].repeat(1, self.config.output_len, 1, 1)
        
        x = torch.cat((x, newx), dim=1) 

        idx = x.reshape(-1, self.config.input_len+self.config.output_len, self.config.J*3)

        return idx

    def forward(self, data): # (B, N, T, J, 3)        
        
        x_list = []
        for n in range(data.shape[1]):
            x = data[:, n].float().clone()
            x_list.append( self.padd( x[:, :self.config.input_len]) )

        out_list, aux = self.m(x_list)
        
        out = torch.cat([_x.unsqueeze(1) for _x in out_list], dim=1)
        
        if self.training:
            return out, aux
        else:
            return out


def create_model(config):
    return MTTPFModel(config).to(config.device)
