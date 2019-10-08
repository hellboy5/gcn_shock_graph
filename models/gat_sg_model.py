"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""
import dgl
import torch
import torch.nn as nn
import dgl.nn.pytorch as conv

class Classifier(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 readout,
                 device):
        super(Classifier, self).__init__()
        self.num_layers = num_layers
        self.layers     = nn.ModuleList()
        self.activation = activation
        self.readout    = readout
        self.device     = device

        # input projection (no residual)
        self.layers.append(conv.GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))

        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(conv.GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.layers.append(conv.GATConv(
            num_hidden * heads[-2], num_hidden, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

        # last layer
        self.classify = nn.Linear(num_hidden, num_classes)
        
    def forward(self,g):

        h=g.ndata['h'].to(self.device)
        for l in range(self.num_layers):
            h = self.layers[l](g, h).flatten(1)

        logits = self.layers[-1](g,h).mean(1)
        g.ndata['h']=logits
                 
        if self.readout == 'max':
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == 'mean':
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.sum_nodes(g,'h')

        return self.classify(hg)
