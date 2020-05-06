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

norm_factors={'rad_scale':1,'angle_scale':1,'length_scale':1,'curve_scale':1,'poly_scale':1}

train_dir='/home/naraym1/stl10/unit_test'

# create train dataset
stl10_1=ShockGraphDataset(train_dir,'stl10',norm_factors,node_app=False,edge_app=False,cache=True,symmetric=True,data_augment=False)


# stl10_2=ShockGraphDataset(train_dir,'stl10',norm_factors,node_app=True,edge_app=False,cache=True,symmetric=True,data_augment=False)
# stl10_3=ShockGraphDataset(train_dir,'stl10',norm_factors,node_app=False,edge_app=True,cache=True,symmetric=True,data_augment=False)
# stl10_4=ShockGraphDataset(train_dir,'stl10',norm_factors,node_app=True,edge_app=True,cache=True,symmetric=True,data_augment=False)

# for i in range(10):
#     g1, l1 = stl10_1[i]
#     g2, l2 = stl10_2[i]
#     g3, l3 = stl10_3[i]
#     g4, l4 = stl10_4[i]

#     h1=g1.ndata['h']
#     h2=g2.ndata['h']
#     h3=g3.ndata['h']
#     h4=g4.ndata['h']

#     print(np.logical_and(h1>-1,h1<1))
#     # print(np.any((h1<-1)|(h1>1)))
#     # print(np.any((h2<-1)|(h2>1)))
#     # print(np.any((h3<-1)|(h3>1)))
#     # print(np.any((h4<-1)|(h4>1)))
    
#     h2=h2[:,:58]
#     h3=np.delete(h3,[38,39,40,51,52,53,64,65,66],axis=1)

#     h4=np.delete(h4,[38,39,40,51,52,53,64,65,66],axis=1)
#     h4=h4[:,:58]

#     print(np.array_equal(h1,h2))
#     print(np.array_equal(h1,h3))
#     print(np.array_equal(h1,h4))
#     print(np.array_equal(h2,h3))
#     print(np.array_equal(h2,h4))
#     print(np.array_equal(h3,h4))


