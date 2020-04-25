import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as conv
from .spp_pooling import SppPooling
from .cov_pooling import CovPooling
from .gcaps_conv import GcapsConv

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes,hidden_layers,in_stats,out_stats,gfc_layers,readout,
                 activation_func,dropout,grid,K,device):
        super(Classifier, self).__init__()
        self.device      = device
        self.readout     = readout
        self.layers      = nn.ModuleList()
        self.batch_norms = nn.ModuleList() 
        self.grid        = grid
        self.K           = K
        self.hidden_dim  = hidden_dim

        self.layers.append(GcapsConv(in_dim,hidden_dim,gfc_layers,1,out_stats,activation=activation_func))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
                
        # hidden layers
        for k in range(0,hidden_layers):
            self.layers.append(GcapsConv(hidden_dim,hidden_dim,gfc_layers,in_stats,out_stats,activation=activation_func))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
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
                nn.Linear(hidden_dim * self.grid * self.grid, n_classes)
                # nn.Conv2d(hidden_dim,64,kernel_size=3,padding=1),
                # nn.ReLU(inplace=True),
                # nn.Dropout(),
                # nn.Linear(2*hidden_dim, 2*hidden_dim),
                # nn.ReLU(inplace=True),
                # nn.Linear(2*hidden_dim, n_classes),
                
                # nn.Conv2d(hidden_dim,64,kernel_size=3,padding=1),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(384,256,kernel_size=3,padding=1),
                #nn.ReLU(inplace=True),
                #nn.Flatten(1),
                #nn.Dropout(p=dropout),
                #nn.Linear(64*self.grid*self.grid, n_classes)
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
            self.classify = nn.Linear(var*out_stats, n_classes)
        
        
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


