import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import dgl
from scipy.ndimage.interpolation import zoom,shift
from data.ShockGraphDataset import ShockGraphDataset

norm_factors={"rad_scale":68.2340,
              "angle_scale":713.8326,
              "length_scale":116.8454,
              "curve_scale":10.817794298253000,
              "poly_scale":8272.887700675255}

#norm_factors={'rad_scale':1,'angle_scale':1,'length_scale':1,'curve_scale':1,'poly_scale':1}

train_dir='/home/naraym1/stl10/unit_train_sift'

# create train dataset
triplet=ShockGraphDataset(train_dir,'stl10',norm_factors,node_app=False,edge_app=False,cache=True,symmetric=True,data_augment=False)


for g in range(1):
    graph,label=triplet[g]

    data=graph.ndata['h']

    print(data.shape)
    norms=torch.norm(data,dim=0)
    print(norms)

    np.savetxt('f_matrix1.txt',data,delimiter=' ')
