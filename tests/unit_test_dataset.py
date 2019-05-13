import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
from scipy.ndimage.interpolation import zoom,shift
from scipy.misc import imread,imresize
from data.ShockGraphDataset import ShockGraphDataset

train_dir='/home/naraym1/stl10/test_code_dir'

def get_pos(graph):
    positions={}
    features=graph.ndata['h']
    for rows in range(0,features.shape[0]):
        positions[rows]=(features[rows,1],features[rows,0])

    return positions

# #Generators
stl10=ShockGraphDataset(train_dir,'stl10',cache=False,symmetric=True,data_augment=True)


fname='/Users/naraym1/work/stl10/se_tcg_train/bird_310-n111-shock_graph.h5'
image='310.png'
graph, label = stl10[0]

trans=stl10.trans
print(trans)
img=imread(image)

if trans[0]:
    img=np.fliplr(img)
    
shifted_input=shift(img, np.array([trans[2][0],trans[2][1],0]))
scaled_input=zoom(shifted_input, np.array([trans[1],trans[1],1]))


print(scaled_input.shape)
positions= get_pos(graph)

fig, ax = plt.subplots()
plt.imshow(scaled_input)
    
nx.draw(graph.to_networkx(), pos=positions,ax=ax)

feature_matrix=graph.ndata['h'];
zero_set=np.array([0.0,0.0])
for row_idx in range(0,feature_matrix.shape[0]):
    pt=feature_matrix[row_idx,9:11]
    plt.plot(pt[1],pt[0],'co')
    
    if np.array_equal(feature_matrix[row_idx,11:13],zero_set)==False:
        pt=feature_matrix[row_idx,11:13]
        plt.plot(pt[1],pt[0],'co')
        
    if np.array_equal(feature_matrix[row_idx,13:15],zero_set)==False:
        pt=feature_matrix[row_idx,13:15]
        plt.plot(pt[1],pt[0],'co')

#ax.set_title('Class: {:d}'.format(label))

plt.plot(48.0,48.0,'g*')
plt.show()












