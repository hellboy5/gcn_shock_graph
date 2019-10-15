import dgl
import torch
import torch.nn as nn
import dgl.nn.pytorch as conv

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes,hidden_layers,aggregate,readout,
                 activation,dropout,device):
        super(Classifier, self).__init__()
        self.device  = device
        self.readout = readout
        self.layers  = nn.ModuleList()
        
        # input layer
        self.layers.append(conv.SAGEConv(in_dim,hidden_dim,aggregate,feat_drop=0.0,
                                    activation=activation))

        # hidden layers
        for k in range(0,hidden_layers):
            self.layers.append(conv.SAGEConv(hidden_dim,hidden_dim,aggregate,feat_drop=dropout,
                                        activation=activation))

        # last layer
        if readout == 'max':
            self.readout_fcn = conv.MaxPooling()
        elif readout == 'mean':
            self.readout_fcn = conv.AvgPooling()
        elif readout == 'sum':
            self.readout_fcn = conv.SumPooling()
        else:
            self.readout_fcn = conv.GlobalAttentionPooling(nn.Linear(hidden_dim,1),nn.Linear(hidden_dim,hidden_dim*2))

        self.classify = nn.Linear(hidden_dim, n_classes)

        
    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        h=g.ndata['h'].to(self.device)
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h

        hg=self.readout_fcn(g,g.ndata['h'])
            
        return self.classify(hg)


