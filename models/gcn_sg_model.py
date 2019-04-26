import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')

def reduce_mean(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}

def reduce_max(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.max(nodes.mailbox['m'], 1)[0]
    return {'h': accum}

def reduce_sum(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.sum(nodes.mailbox['m'], 1)
    return {'h': accum}

class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation,aggregate):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

        if aggregate == 'max':
            self.reduce = reduce_max
        elif aggregate == 'mean':
            self.reduce = reduce_mean
        else:
            self.reduce = reduce_sum

    def forward(self, g, feature):
        # Initialize the node features with h.
        g.ndata['h'] = feature
        g.update_all(msg, self.reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes,hidden_layers,aggregate,combine,device):
        super(Classifier, self).__init__()
        self.device = device
        self.combine = combine

        self.layers = nn.ModuleList()
        
        # input layer
        self.layers.append(GCN(in_dim, hidden_dim, F.relu,aggregate))

        # hidden layers
        for k in range(0,hidden_layers):
            self.layers.append(GCN(hidden_dim, hidden_dim, F.relu,aggregate))

        # last layer
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        h=g.ndata['h'].to(self.device)
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        
        if self.combine == 'max':
            hg = dgl.max_nodes(g, 'h')
        elif self.combine == 'mean':
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.sum_nodes(g,'h')
            
        return self.classify(hg)


