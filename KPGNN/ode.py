from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq
import math
import dgl.function as fn
class MultiHeadGATODE(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, activation, dropout=0.9, bias=True, merge='cat', odetype="gcn"):
        super(MultiHeadGATODE, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            if odetype == "gat":
                self.heads.append(GATODE(in_dim, out_dim, activation))
            else:
                self.heads.append(GCNODE(g, in_dim, out_dim, activation, dropout, bias))
        self.merge = merge
        self.odetype = odetype

    def forward(self, t, h):
        if self.odetype == "gat":
            head_outs = [attn_head(h) for attn_head in self.heads]
        else:
            head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))

class GCNODE(nn.Module):
    def __init__(self, g, in_feats, out_feats, activation, dropout=0.9, bias=True):
        super(GCNODE, self).__init__()
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.g = g
        self.weight = nn.Linear(in_feats, out_feats, bias)
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.reset_parameters(bias)
    
    def reset_parameters(self, bias=True):
        """Reinitialize learnable parameters."""
        stdv = 1. / math.sqrt(self.weight.weight.size(1))
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        if bias:
            nn.init.uniform_(self.weight.bias, -stdv, stdv)
        
    def forward(self, h):
        '''
        GCN
        '''
        h = self.weight(h) 
        h = h * self.nf.layers[self.layer_id].data["norm"]
        self.nf.layers[self.layer_id].data['h'] = h 
        self.nf.block_compute(self.layer_id,  # block_id _ The block to run the computation.
                         fn.copy_u(u='h', out='m'),fn.sum(msg='m', out='h'))  # Reduce function on the node.
        
        h = self.nf.layers[self.layer_id].data.pop('h')
        h = h * self.nf.layers[self.layer_id].data["norm"]
        if self.activation:
            h = self.activation(h)
        if self.dropout:
            h = self.dropout(h)
        self.nf.layers[self.layer_id].data['h'] = h
        return h