import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import networkx as nx
from data.ShockGraphDataset import ShockGraphDataset

train_dir='/home/naraym1/cifar_100/train_dir'
label_file='/home/naraym1/cifar_100/scripts/labels.txt'

def get_pos(graph):
    positions={}
    features=graph.ndata['h'];
    for rows in range(0,features.shape[0]):
        positions[rows]=(features[rows,0],features[rows,1])

    return positions

# #Generators
dataset=ShockGraphDataset(train_dir,label_file,symmetric=False,shuffle=True)

for i in range(0,100):
    graph, label = dataset[i]
    positions= get_pos(graph)
    fig, ax = plt.subplots()
    nx.draw(graph.to_networkx(), pos=positions,ax=ax)
    ax.set_title('Class: {:d}'.format(label))
    plt.show()













