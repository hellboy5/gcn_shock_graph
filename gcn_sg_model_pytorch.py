import dgl
import dgl.function as fn
import torch
import json
import sys
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from data.ShockGraphDataset import ShockGraphDataset
import torch.optim as optim
from torch.utils.data import DataLoader

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels).to(device)

# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')

def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.mean(nodes.mailbox['m'], 1)
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
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        # Initialize the node features with h.
        g.ndata['h'] = feature
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()

        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu)])
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        h=g.ndata['h'].to(device)
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)


def train(train_dir,dataset,cache_io,symm_io,shuffle_io,num_classes,input_dim):
    epochs=400
    hidden_dim=256
    batch_io=64
    
    trainset=ShockGraphDataset(train_dir,dataset,cache=cache_io,symmetric=symm_io,shuffle=shuffle_io)

    # Use PyTorch's DataLoader and the collate function
    # defined before.
    data_loader = DataLoader(trainset, batch_size=batch_io, shuffle=True,
                             collate_fn=collate)
    
    # Create model
    model = Classifier(input_dim, hidden_dim, num_classes)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    model.train()

    epoch_losses = []
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for iter, (bg, label) in enumerate(data_loader):
            prediction = model(bg)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        epoch_losses.append(epoch_loss)
        print('Epoch {}, loss {:.6f}'.format(epoch, epoch_loss))

        if (epoch+1) % 50 == 0:
            path='gcn_sg_model_'+dataset+'_epoch_'+str(epoch+1).zfill(3)+'.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss}, path)

        





if __name__ == "__main__":

    global device
    
    cfg=sys.argv[1]
    config_file=json.load(open(cfg))
    train_dir=config_file['train_dir']
    dataset=config_file['dataset']
    cache_io=config_file['cache_io']
    symm_io=config_file['symm_io']
    shuffle_io=config_file['shuffle_io']
    num_classes=config_file['num_classes']
    input_dim=config_file['features_dim']

    device_id=sys.argv[2]
    device="cuda:"+str(device_id)
    train(train_dir,dataset,cache_io,symm_io,shuffle_io,num_classes,input_dim)
    
