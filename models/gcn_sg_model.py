import dgl
import torch
import torch.nn as nn
import dgl.nn.pytorch as conv

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes,hidden_layers,readout,
                 activation_func,dropout,device):
        super(Classifier, self).__init__()
        self.device  = device
        self.readout = readout
        self.layers  = nn.ModuleList()
        
        self.layers.append(conv.GraphConv(in_dim,hidden_dim,activation=activation_func))
            
        # hidden layers
        for k in range(0,hidden_layers):
            self.layers.append(conv.GraphConv(hidden_dim,hidden_dim,activation=activation_func))

        # last layer
        self.dropout  = nn.Dropout(p=dropout)
        self.classify = nn.Linear(hidden_dim, n_classes)
        
        
    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        h=g.ndata['h'].to(self.device)

        for i, conv in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = conv(g, h)
            
        g.ndata['h'] = h

        if self.readout == 'max':
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == 'mean':
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.sum_nodes(g,'h')
            
        return self.classify(hg)


