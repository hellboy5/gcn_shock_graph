import dgl
import torch
import torch.nn as nn
import dgl.nn.pytorch as conv
from .spp_pooling import SppPooling

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes,hidden_layers,ctype,hops,readout,
                 activation_func,dropout,device,grid=8):
        super(Classifier, self).__init__()
        self.device  = device
        self.readout = readout
        self.layers  = nn.ModuleList()
        self.grid    = grid
        
        # input layer
        if ctype == 'tagconv':
            self.layers.append(conv.TAGConv(in_dim,hidden_dim,hops,activation=activation_func))
        else:
            self.layers.append(conv.SGConv(in_dim,hidden_dim,hops,cached=False,norm=activation_func))
            
        # hidden layers
        for k in range(0,hidden_layers):
            if ctype == 'tagconv':
                self.layers.append(conv.TAGConv(hidden_dim,hidden_dim,hops,activation=activation_func))
            else:
                self.layers.append(conv.SGConv(hidden_dim,hidden_dim,hops,cached=False,norm=activation_func))

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
                nn.Linear(2*hidden_dim, n_classes),
            )
        else:
            var=hidden_dim
            if self.readout == 'gap':
                var*=2
            self.classify = nn.Linear(var, n_classes)
        
        
    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        h=g.ndata['h'].to(self.device)

        for i, conv in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = conv(g, h)
            
        g.ndata['h'] = h

        
        if self.readout == 'spp':
            hg=self.readout_fcn(g,g.ndata['h'],g.ndata['x'])
            hg=torch.flatten(hg,1)
        else:
            hg=self.readout_fcn(g,g.ndata['h'])

        return self.classify(hg)


