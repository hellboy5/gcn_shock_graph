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
from data.ShockGraphDataset_fsl import ShockGraphDataset

norm_factors={"rad_scale":68.2340,
              "angle_scale":713.8326,
              "length_scale":116.8454,
              "curve_scale":10.817794298253000,
              "poly_scale":8272.887700675255}

norm_factors={'rad_scale':1,'angle_scale':1,'length_scale':1,'curve_scale':1,'poly_scale':1}

train_dir='/home/naraym1/mini_imagenet/se_tcg_test'

# create train dataset
triplet=ShockGraphDataset(train_dir,'mignt',norm_factors,n_shot=5,k_way=5,episodes=10,test_samples=15,node_app=False,edge_app=False,cache=True,symmetric=True,data_augment=False)


for g in range(1):
    value=triplet[g]
