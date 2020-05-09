import dgl
import torch
import torch.nn as nn
import dgl.nn.pytorch as conv
import torch.nn.functional as F
from .spp_pooling import SppPooling
from .cov_pooling import CovPooling

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim,embed_dim,hidden_layers,hops,readout,
                 activation_func,dropout,local,norm,grid,K,device):
        super(Classifier, self).__init__()
        self.device      = device
        self.readout     = readout
        self.layers      = nn.ModuleList()
        self.batch_norms = nn.ModuleList() 
        self.grid        = grid
        self.K           = K
        self.hidden_dim  = hidden_dim
        self.local       = local
        self.norm        = norm
        
        self.layers.append(conv.TAGConv(in_dim,hidden_dim,hops,activation=activation_func))
                
        # hidden layers
        for k in range(0,hidden_layers):
            self.layers.append(conv.TAGConv(hidden_dim,hidden_dim,hops,activation=activation_func))
            
        # dropout layer
        self.dropout=nn.Dropout(p=dropout)

        if self.local:
            return
        
        # readout layer
        if self.readout == 'max':
            self.readout_fcn = conv.MaxPooling()
        elif self.readout == 'mean':
            self.readout_fcn = conv.AvgPooling()
        elif self.readout == 'sum':
            self.readout_fcn = conv.SumPooling()
        elif self.readout == 'gap':
            self.readout_fcn = conv.GlobalAttentionPooling(nn.Linear(hidden_dim,1),nn.Linear(hidden_dim,hidden_dim*2))
        elif self.readout == 'sort':
            self.readout_fcn = conv.SortPooling(self.K)
        elif self.readout == 'set':
            self.readout_fcn = conv.Set2Set(hidden_dim,2,1)
        elif self.readout == 'cov':
            self.readout_fcn = CovPooling(hidden_dim)
        else:
            self.readout_fcn = SppPooling(hidden_dim,self.grid)
        
        if self.readout == 'spp':
            self.embed = nn.Sequential(
                nn.Dropout(),
                nn.Linear(hidden_dim * self.grid * self.grid, hidden_dim*2),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(2*hidden_dim, 2*hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(2*hidden_dim, embed_dim)
            )
        elif self.readout == 'sort':
            self.embed = nn.Sequential(
                #nn.Dropout(),
                nn.Linear(hidden_dim*self.K, embed_dim)
            )
        elif self.readout == 'cov':
            self.embed = nn.Sequential(
                nn.Dropout(),
                nn.Linear( int(((hidden_dim+1)*hidden_dim)/2), embed_dim)
            )            
        else:
            var=hidden_dim
            if self.readout == 'gap' or self.readout == 'set':
                var*=2
            self.embed = nn.Linear(var, embed_dim)
                
        
    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        h=g.ndata['h'].to(self.device)

        for idx in range(len(self.layers)):
            if idx != 0:
                h = self.dropout(h)

            h = self.layers[idx](g,h)
            
        g.ndata['h'] = h

        # extract features from each node
        if self.local:
            hg=g.ndata['h']
        #extract global features from graph
        else:
            if self.readout == 'spp':
                hg=self.readout_fcn(g,g.ndata['h'],g.ndata['x'])
                hg=torch.flatten(hg,1)
            else:
                hg=self.readout_fcn(g,g.ndata['h'])
                
            hg=self.embed(hg)
            
        if self.norm:
            hg=F.normalize(hg)

        return hg


