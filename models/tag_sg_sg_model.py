import dgl
import torch
import torch.nn as nn
import dgl.nn.pytorch as conv
import torch.nn.functional as F
from .spp_pooling import SppPooling
from .cov_pooling import CovPooling
from .graph_norm import GraphNorm
from .compute_hist_features import HistFeatures

#Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
#https://arxiv.org/abs/1908.08681v1
#implemented for PyTorch / FastAI by lessw2020
#github: https://github.com/lessw2020/mish

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(F.softplus(x)))
        
class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes,hidden_layers,ctype,hops,readout,
                 activation_func,dropout,grid,K,norm,device):
        super(Classifier, self).__init__()
        self.device      = device
        self.readout     = readout
        self.layers      = nn.ModuleList()
        self.n_layers    = nn.ModuleList() 
        self.grid        = grid
        self.K           = K
        self.hidden_dim  = hidden_dim
        self.norm        = norm

        self.mish = Mish()
        
        # input layer
        if ctype == 'tagconv':
            self.layers.append(conv.TAGConv(in_dim,hidden_dim,hops,activation=activation_func))
        else:
            self.layers.append(conv.SGConv(in_dim,hidden_dim,hops,cached=False,norm=activation_func))

        if self.norm == 'batch':
            self.n_layers.append(nn.BatchNorm1d(hidden_dim))
        elif self.norm == 'layer':
            self.n_layers.append(nn.LayerNorm(hidden_dim,elementwise_affine=False))
        elif self.norm == 'group':
            self.n_layers.append(nn.GroupNorm(16,hidden_dim))
        elif self.norm == 'instance':
            self.n_layers.append(nn.InstanceNorm1d(hidden_dim))
        else:
            self.n_layers.append(GraphNorm(hidden_dim,affine=False))
            
        # hidden layers
        for k in range(0,hidden_layers):
            if ctype == 'tagconv':
                self.layers.append(conv.TAGConv(hidden_dim,hidden_dim,hops,activation=activation_func))
            else:
                self.layers.append(conv.SGConv(hidden_dim,hidden_dim,hops,cached=False,norm=activation_func))

            if self.norm == 'batch':
                self.n_layers.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm == 'layer':
                self.n_layers.append(nn.LayerNorm(hidden_dim,elementwise_affine=False))
            elif self.norm == 'group':
                self.n_layers.append(nn.GroupNorm(16,hidden_dim))
            elif self.norm == 'instance':
                self.n_layers.append(nn.InstanceNorm1d(hidden_dim))
            else:
                self.n_layers.append(GraphNorm(hidden_dim,affine=False))
        
            
        # dropout layer
        self.dropout=nn.Dropout(p=dropout)
                
        # last layer
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
            self.classify = nn.Sequential(
                nn.Dropout(),
                nn.Linear(hidden_dim * self.grid * self.grid, hidden_dim*2),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(2*hidden_dim, 2*hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(2*hidden_dim, n_classes)
            )
        elif self.readout == 'sort':
            self.classify = nn.Sequential(
                #nn.Dropout(),
                nn.Linear(hidden_dim*self.K, n_classes)
            )
        elif self.readout == 'cov':
            self.classify = nn.Sequential(
                nn.Dropout(),
                nn.Linear( int(((hidden_dim+1)*hidden_dim)/2), n_classes)
            )            
        else:
            var=hidden_dim
            if self.readout == 'gap' or self.readout == 'set':
                var*=2
            self.classify = nn.Linear(var, n_classes)
        
        
    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        h=g.ndata['h'].to(self.device)

        for idx in range(len(self.layers)):
            if idx != 0:
                h = self.dropout(h)

            h = self.layers[idx](g,h)
            
        g.ndata['h'] = h

        
        if self.readout == 'spp':
            hg=self.readout_fcn(g,g.ndata['h'],g.ndata['x'])
            hg=torch.flatten(hg,1)
        else:
            hg=self.readout_fcn(g,g.ndata['h'])

        return self.classify(hg)


