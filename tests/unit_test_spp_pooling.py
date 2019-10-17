import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import dgl
from scipy.ndimage.interpolation import zoom,shift
from scipy.misc import imread,imresize
from data.ShockGraphDataset import ShockGraphDataset
from models.spp_pooling import SppPooling


def write_slice(pool_results,fname):
    # Write the array to disk
    with open(fname, 'w') as outfile:
        for idx in range(pool_results.shape[2]):
        
            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            np.savetxt(outfile, pool_results[:,:,idx], delimiter=' ')


train_dir='/home/naraym1/stl10/unit_test'
# #Generators
ngrid=8
stl10=ShockGraphDataset(train_dir,'stl10',cache=True,symmetric=True,data_augment=False,grid=ngrid)

print(stl10.files)
g1, l1 = stl10[0]
g2, l2 = stl10[1]
g3, l3 = stl10[2]


bg = dgl.batch([g1, g2, g3])


spp_pool=SppPooling(58,ngrid)

hg=spp_pool(bg,bg.ndata['h'],bg.ndata['x'])
print(bg.batch_size)
print(bg.batch_num_nodes)
print(hg.shape)

write_slice(hg[0,:,:,:],'vm_spp1.txt')
write_slice(hg[1,:,:,:],'vm_spp2.txt')
write_slice(hg[2,:,:,:],'vm_spp3.txt')

# feature_matrix=graph.ndata['h'];
# print(np.any(feature_matrix>1))
# print(np.any(feature_matrix<-1))

np.savetxt('f_matrix1.txt',g1.ndata['h'],delimiter=' ')
np.savetxt('f_matrix2.txt',g2.ndata['h'],delimiter=' ')
np.savetxt('f_matrix3.txt',g3.ndata['h'],delimiter=' ')







