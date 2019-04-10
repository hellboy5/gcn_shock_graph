import numpy as np
import glob
import random
import h5py
import os
import re
import dgl
import torch

from tqdm import tqdm
from torch.utils.data.dataset import Dataset


class ShockGraphDataset(Dataset):
    'Generates data for Keras'
    def __init__(self,directory,labels,cache=True,symmetric=False,shuffle=True):
        'Initialization'
        
        self.directory = directory
        self.labels = labels
        self.cache=cache
        self.symmetric = symmetric
        self.shuffle = shuffle
        self.files=[]
        self.class_mapping={}
        self.sg_graphs=[]
        self.sg_labels=[]
        
        self.__gen_file_list()
        self.__parse_label()
        
        if self.cache:
            self.__preprocess_graphs()
            

        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.files)

    def __getitem__(self, index):
        'Generate one example of data'

        if self.cache:
            graph=self.sg_graphs[index]
            label=self.sg_labels[index]
        else:        
            item=self.files[index]
            graph=self.__read_shock_graph(item)
            obj=os.path.basename(item)
            obj=re.sub("_to_msel.*","",obj)
            class_name=obj[:obj.rfind('_')]
            label=self.class_mapping[class_name]

        return graph,label

    def __preprocess_adj_numpy(self,adj, symmetric=True):
        adj = adj + np.eye(adj.shape[0])
        adj = self.__normalize_adj_numpy(adj, symmetric)
        return adj

    def __preprocess_adj_numpy_with_identity(self,adj, symmetric=True):
        adj = adj + np.eye(adj.shape[0])
        adj = self.__normalize_adj_numpy(adj, symmetric)
        adj = np.concatenate([np.eye(adj.shape[0]), adj], axis=0)
        return adj

    def __normalize_adj_numpy(self,adj, symmetric=True):
        if symmetric:
            d = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
            a_norm = adj.dot(d).transpose().dot(d)
        else:
            d = np.diag(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
            a_norm = d.dot(adj)
        return a_norm
        
    def __gen_file_list(self):
        self.files=glob.glob(self.directory+'/*.h5')
        if self.shuffle:
            random.shuffle(self.files)

    def __parse_label(self):
        file=open(self.labels,'r')
        lines= file.read().splitlines() 
        for idx in range(0,len(lines)):
            self.class_mapping[lines[idx]]=idx

    def __preprocess_graphs(self):

        for fid in tqdm(self.files):
            graph=self.__read_shock_graph(fid)

            obj=os.path.basename(fid)
            obj=re.sub("_to_msel.*","",obj)
            class_name=obj[:obj.rfind('_')]
            label=self.class_mapping[class_name]

            self.sg_graphs.append(graph)
            self.sg_labels.append(label)

        
    def __read_shock_graph(self,sg_file):
        fid=h5py.File(sg_file,'r')

        # read in features
        feature_data=fid.get('feature')
        F_matrix=np.array(feature_data).astype(np.float32)
        
        # read in adj matrix
        adj_data=fid.get('adj_matrix')
        adj_matrix=np.array(adj_data)
    
        # convert to dgl
        G=dgl.DGLGraph()
        G.add_nodes(adj_matrix.shape[0])
        G.set_n_initializer(dgl.init.zero_initializer)
 
        for row in range(0,adj_matrix.shape[0]):
            neighbors=adj_matrix[row,:]
            target=np.nonzero(neighbors)[0]
            source=np.zeros(target.shape)+row

            if target.size:
                G.add_edges(source,target)
                if self.symmetric:
                    G.add_edges(target,source)
                    
            F=torch.from_numpy(F_matrix[row,:])
            feature=F.view(-1,F.shape[0])
            G.nodes[row].data['h']=feature
                    
        return G

