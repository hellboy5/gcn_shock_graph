"""
APPNP implementation in DGL.
References
----------
Paper: https://arxiv.org/abs/1810.05997
Author's code: https://github.com/klicperajo/ppnp
"""
import dgl
import torch
import torch.nn as nn
import dgl.nn.pytorch as conv
import torch.nn.functional as F
from .spp_pooling import SppPooling

class Classifier(nn.Module):
    def __init__(self,in_dim,hidden_dim,n_classes,hidden_layers,readout,
                 activation,feat_drop,edge_drop,alpha,K,grid,device):
        super(Classifier, self).__init__()
        self.device = device
        self.readout = readout
        self.layers = nn.ModuleList()
        self.grid = grid
        # input layer
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        # hidden layers
        for i in range(hidden_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.activation = activation

        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x

        self.propagate = conv.APPNPConv(K, alpha, edge_drop)

        # last layer
        if self.readout == 'max':
            self.readout_fcn = conv.MaxPooling()
        elif self.readout == 'mean':
            self.readout_fcn = conv.AvgPooling()
        elif self.readout == 'sum':
            self.readout_fcn = conv.SumPooling()
        elif self.readout == 'gap':
            self.readout_fcn = conv.GlobalAttentionPooling(nn.Linear(hidden_dim,1),nn.Linear(hidden_dim,hidden_dim*2))
        else:
            self.readout_fcn = SppPooling(hidden_dim,self.grid)
        
        if self.readout == 'spp':
            self.classify = nn.Sequential(
                nn.Dropout(),
                nn.Linear(hidden_dim * self.grid * self.grid, hidden_dim*2),
                nn.ReLU(inplace=True),
                nn.Linear(2*hidden_dim, n_classes),
            )
        else:
            var=hidden_dim
            if self.readout == 'gap':
                var*=2
            self.classify = nn.Linear(var, n_classes)


        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g):
        # prediction step
        h = g.ndata['h'].to(self.device)

        for idx in range(len(self.layers)):
            if idx != 0:
                h = self.feat_drop(h)

            h = self.activation(self.layers[idx](h))
            
        g.ndata['h'] = h

        # propagation step
        h = self.propagate(g, h)
        
        g.ndata['h'] = h

        if self.readout == 'spp':
            hg=self.readout_fcn(g,g.ndata['h'],g.ndata['x'])
            hg=torch.flatten(hg,1)
        else:
            hg=self.readout_fcn(g,g.ndata['h'])

        return self.classify(hg)
