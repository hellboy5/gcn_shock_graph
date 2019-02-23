import numpy as np
import h5py
import sys
import math
import glob
import h5py
import os

from collections import defaultdict
from scipy.sparse import coo_matrix

def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))


def read_shock_graph(sg_file):
    fid=h5py.File(sg_file,'r')
    
    # read in features
    feature_data=fid.get('feature')
    F_matrix=np.array(feature_data)

    # read in adj matrix
    adj_data=fid.get('adj_matrix')
    adj_matrix=np.array(adj_data)

    # read in debug info
    debug_data=fid.get('debug')
    debug_matrix=np.array(debug_data)

    ref_pt=debug_matrix[:2]
    max_offsets=debug_matrix[2:4]
    max_radius=debug_matrix[4]
    
    xdata=F_matrix[:,1]*max_offsets[1]+ref_pt[1]
    ydata=F_matrix[:,0]*max_offsets[0]+ref_pt[0]

    xmin=round_nearest(np.min(xdata),0.5)
    xmax=round_nearest(np.max(xdata),0.5)


    if xmin != -16:
        print sg_file,'BIG ERROR'
        return
    
    if xmax != 48:
        print sg_file,'BIG ERROR'
        return
    
    grid=[ round_nearest(val,0.5) for val in np.linspace(xmin,xmax,(xmax-xmin)*2+1)]
    
    idx=0
    mapping_location=dict()
    for ygrid in grid:
        for xgrid in grid:
            mapping_location[(ygrid,xgrid)]=idx
            idx=idx+1
            
    pooled_adj_matrix=np.zeros((idx,idx),dtype='uint8')
    pooled_feature_matrix=np.zeros((idx,F_matrix.shape[1]))
    
    nodes_to_pooled_node=dict()

    orig_to_key_mapping=defaultdict(list)
    
    visited=set()
    row_to_degree_mapping={}
    
    for ii in range(0,len(xdata)):
        adj_row=adj_matrix[ii,:]
        degree=np.sum(adj_row)+np.sum(adj_matrix[:,ii])
        neighbors=np.nonzero(adj_row)
        row_key=(round_nearest(ydata[ii],0.5),round_nearest(xdata[ii],0.5))
        orig_to_key_mapping[row_key].append((ydata[ii],xdata[ii]))
        row_idx=mapping_location[row_key]
        
        
        if row_idx not in row_to_degree_mapping:
            pooled_feature_matrix[row_idx,:]=F_matrix[ii,:]
            row_to_degree_mapping[row_idx]=degree
        else:
            print 'Revisiting node'
            print (row_idx,ii,ydata[ii],xdata[ii],row_key)
            if degree > row_to_degree_mapping[row_idx]:
                pooled_feature_matrix[row_idx,:]=F_matrix[ii,:]
                row_to_degree_mapping[row_idx]=degree
        
        for val in neighbors[0]:
            x_neigh=F_matrix[val,1]*max_offsets[1]+ref_pt[1]
            y_neigh=F_matrix[val,0]*max_offsets[0]+ref_pt[0]
            key=(round_nearest(y_neigh,0.5),round_nearest(x_neigh,0.5))
            adj_location=mapping_location[key]
            pooled_adj_matrix[row_idx,adj_location]=1


        
    sparse_adj_matrix=coo_matrix(pooled_adj_matrix)
    sparse_feature_matrix=coo_matrix(pooled_feature_matrix)

    adj_indices=zip(sparse_adj_matrix.row,sparse_adj_matrix.col)
    feature_indices=zip(sparse_feature_matrix.row,sparse_feature_matrix.col)

    fname=os.path.dirname(sg_file)+'/'+os.path.basename(sg_file).split('.')[0]+'_reordered.h5'
    print 'Writing out to',fname
    hf = h5py.File(fname,'w')

    hf.create_dataset('sparse_adj_data',data=sparse_adj_matrix.data)
    hf.create_dataset('sparse_adj_indices',data=adj_indices)
    hf.create_dataset('sparse_adj_shape',data=sparse_adj_matrix.shape)


    hf.create_dataset('sparse_feature_data',data=sparse_feature_matrix.data)
    hf.create_dataset('sparse_feature_indices',data=feature_indices)
    hf.create_dataset('sparse_feature_shape',data=sparse_feature_matrix.shape)

    hf.close()

            
if __name__ == '__main__':

    path=sys.argv[1]
    files=glob.glob(path+'/*.h5')
    for f in files:
        print 'Working on ',f
        read_shock_graph(f)

    

    
