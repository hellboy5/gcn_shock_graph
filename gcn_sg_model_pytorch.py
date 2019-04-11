import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from data.ShockGraphDataset import ShockGraphDataset
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


# Create training and test sets.
train_dir='/home/naraym1/cifar_100/train_dir'
test_dir='/home/naraym1/cifar_100/test_dir'
label_file='/home/naraym1/cifar_100/scripts/labels.txt'
num_classes=100
epochs=200

trainset=ShockGraphDataset(train_dir,label_file,cache=False,symmetric=False,shuffle=True)
testset=ShockGraphDataset(test_dir,label_file,cache=False,symmetric=False,shuffle=True)

# Use PyTorch's DataLoader and the collate function
# defined before.
data_loader = DataLoader(trainset, batch_size=32, shuffle=True,
                         collate_fn=collate)

# Create model
model = Classifier(19, 256, num_classes)
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

    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))

    if epoch % 50 == 0:
        path='gcn_sg_model_epoch_'+str(epoch).zfill(3)+'.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, path)

    epoch_losses.append(epoch_loss)

###############################################################################
# The learning curve of a run is presented below:

#plt.title('cross entropy averaged over minibatches')
#plt.plot(epoch_losses)
#plt.show()

###############################################################################
# The trained model is evaluated on the test set created. Note that for deployment
# of the tutorial, we restrict our running time and you are likely to get a higher
# accuracy (:math:`80` % ~ :math:`90` %) than the ones printed below.

model.eval()
# Convert a list of tuples to two lists
test_X, test_Y = map(list, zip(*testset))
test_bg = dgl.batch(test_X)
test_Y = torch.tensor(test_Y).float().view(-1, 1)
probs_Y = torch.softmax(model(test_bg), 1)
sampled_Y = torch.multinomial(probs_Y, 1)
argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
    (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
    (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))

###############################################################################
# Below is an animation where we plot graphs with the probability a trained model
# assigns its ground truth label to it:
#
# .. image:: https://s3.us-east-2.amazonaws.com/dgl.ai/tutorial/batch/test_eval4.gif
#
# To understand the node/graph representations a trained model learnt,
# we use `t-SNE, <https://lvdmaaten.github.io/tsne/>`_ for dimensionality reduction
# and visualization.
#
# .. image:: https://s3.us-east-2.amazonaws.com/dgl.ai/tutorial/batch/tsne_node2.png
#     :align: center
#
# .. image:: https://s3.us-east-2.amazonaws.com/dgl.ai/tutorial/batch/tsne_graph2.png
#     :align: center
#
# The two small figures on the top separately visualize node representations after :math:`1`,
# :math:`2` layers of graph convolution and the figure on the bottom visualizes
# the pre-softmax logits for graphs as graph representations.
#
# While the visualization does suggest some clustering effects of the node features,
# it is expected not to be a perfect result as node degrees are deterministic for
# our node features. Meanwhile, the graph features are way better separated.
#
# What's Next?
# ------------
# Graph classification with graph neural networks is still a very young field
# waiting for folks to bring more exciting discoveries! It is not easy as it
# requires mapping different graphs to different embeddings while preserving
# their structural similarity in the embedding space. To learn more about it,
# `"How Powerful Are Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_
# in ICLR 2019 might be a good starting point.
#
# With regards to more examples on batched graph processing, see
#
# * our tutorials on `Tree LSTM <https://docs.dgl.ai/tutorials/models/2_small_graph/3_tree-lstm.html>`_ and `Deep Generative Models of Graphs <https://docs.dgl.ai/tutorials/models/3_generative_model/5_dgmg.html>`_
# * an example implementation of `Junction Tree VAE <https://github.com/dmlc/dgl/tree/master/examples/pytorch/jtnn>`_
