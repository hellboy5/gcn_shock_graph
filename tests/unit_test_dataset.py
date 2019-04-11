import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import networkx as nx
from data.ShockGraphDataset import ShockGraphDataset

cifar100_train_dir='/home/naraym1/cifar_100/train_dir'
stl10_train_dir='/home/naraym1/stl10/se_tcg_train'

def get_pos(graph):
    positions={}
    features=graph.ndata['h'];
    for rows in range(0,features.shape[0]):
        positions[rows]=(features[rows,0],features[rows,1])

    return positions

# #Generators
cifar100=ShockGraphDataset(cifar100_train_dir,'cifar100',cache=False,symmetric=False,shuffle=True)
stl10=ShockGraphDataset(stl10_train_dir,'stl10',cache=False,symmetric=False,shuffle=True)

for i in range(0,5):
    graph, label = cifar100[i]
    positions= get_pos(graph)
    fig, ax = plt.subplots()
    nx.draw(graph.to_networkx(), pos=positions,ax=ax)
    ax.set_title('Class: {:d}'.format(label))
    plt.show()

for i in range(0,5):
    graph, label = stl10[i]
    positions= get_pos(graph)
    fig, ax = plt.subplots()
    nx.draw(graph.to_networkx(), pos=positions,ax=ax)
    ax.set_title('Class: {:d}'.format(label))
    plt.show()












