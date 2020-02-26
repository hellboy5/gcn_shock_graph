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
from scipy.misc import imread,imresize
from data.ShockGraphDataset import ShockGraphDataset
from models.cov_pooling import CovPooling


def write_slice(pool_results,fname):
    # Write the array to disk
    with open(fname, 'w') as outfile:
        for idx in range(pool_results.shape[2]):
        
            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            np.savetxt(outfile, pool_results[:,:,idx], delimiter=' ')





norm_factors={'rad_scale':1,'angle_scale':1,'length_scale':1,'curve_scale':1,'poly_scale':1}

train_dir='/home/naraym1/stl10/small_set'

# create train dataset
stl10=ShockGraphDataset(train_dir,'stl10',norm_factors,app=False,cache=True,symmetric=True,data_augment=False)

print(stl10.files)
g1, l1 = stl10[0]
g2, l2 = stl10[1]
g3, l3 = stl10[2]


bg = dgl.batch([g1, g2, g3])


cov_pool=CovPooling(58)

hg=cov_pool(bg,bg.ndata['h'])
print(bg.batch_size)
print(bg.batch_num_nodes)
print(hg.shape)
#print(np.sum(bg.batch_num_edges))

np.savetxt('vm_cov.txt',hg,delimiter=' ')


# # feature_matrix=graph.ndata['h'];
# # print(np.any(feature_matrix>1))
# # print(np.any(feature_matrix<-1))

np.savetxt('f_matrix1.txt',g1.ndata['h'],delimiter=' ')
np.savetxt('f_matrix2.txt',g2.ndata['h'],delimiter=' ')
np.savetxt('f_matrix3.txt',g3.ndata['h'],delimiter=' ')







