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


stl10_map={'airplane':0, 'bird':1,'car':2,'cat':3,'deer':4,'dog':5,'horse':6,'monkey':7,'ship':8,'truck':9}

cifar100_map={'couch': 25, 'pine_tree': 59, 'butterfly': 14, 'mountain': 49, 'bus': 13, 'sunflower': 82, 'beetle': 7, 'lamp': 40, 'can': 16, 'beaver': 4, 'bee': 6, 'elephant': 31, 'kangaroo': 38, 'oak_tree': 52, 'orchid': 54, 'trout': 91, 'pickup_truck': 58, 'chimpanzee': 21, 'worm': 99, 'snail': 77, 'television': 87, 'hamster': 36, 'squirrel': 80, 'lion': 43, 'otter': 55, 'shrew': 74, 'pear': 57, 'turtle': 93, 'seal': 72, 'plate': 61, 'fox': 34, 'train': 90, 'porcupine': 63, 'tulip': 92, 'keyboard': 39, 'orange': 53, 'possum': 64, 'cattle': 19, 'skyscraper': 76, 'bear': 3, 'cup': 28, 'cockroach': 24, 'caterpillar': 18, 'mouse': 50, 'forest': 33, 'rabbit': 65, 'aquarium_fish': 1, 'chair': 20, 'castle': 17, 'palm_tree': 56, 'telephone': 86, 'mushroom': 51, 'streetcar': 81, 'willow_tree': 96, 'man': 46, 'wardrobe': 94, 'bowl': 10, 'sweet_pepper': 83, 'maple_tree': 47, 'snake': 78, 'whale': 95, 'poppy': 62, 'tank': 85, 'bed': 5, 'rose': 70, 'crocodile': 27, 'raccoon': 66, 'tractor': 89, 'bicycle': 8, 'bridge': 12, 'dinosaur': 29, 'crab': 26, 'clock': 22, 'bottle': 9, 'lawn_mower': 41, 'road': 68, 'spider': 79, 'skunk': 75, 'tiger': 88, 'sea': 71, 'lizard': 44, 'cloud': 23, 'ray': 67, 'house': 37, 'lobster': 45, 'boy': 11, 'plain': 60, 'table': 84, 'dolphin': 30, 'camel': 15, 'rocket': 69, 'baby': 2, 'girl': 35, 'shark': 73, 'motorcycle': 48, 'flatfish': 32, 'leopard': 42, 'wolf': 97, 'apple': 0, 'woman': 98}

class ShockGraphDataset(Dataset):
    'Generates data for Keras'
    def __init__(self,directory,dataset,cache=True,symmetric=False):
        'Initialization'
        
        self.directory = directory
        self.cache=cache
        self.symmetric = symmetric
        self.files=[]
        self.class_mapping={}
        self.sg_graphs=[]
        self.sg_labels=[]

        if dataset=='cifar100':
            print('Using cifar 100 dataset')
            self.class_mapping=cifar100_map
        else:
            print('Using stl 10 dataset')
            self.class_mapping=stl10_map
            
        self.__gen_file_list()
        
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
            obj=re.split(r'[0-9].*',obj)[0]
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

    def __preprocess_graphs(self):

        for fid in tqdm(self.files):
            graph=self.__read_shock_graph(fid)

            obj=os.path.basename(fid)
            obj=re.split(r'[0-9].*',obj)[0]
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

def collate(samples,device_name):
        # The input `samples` is a list of pairs
        #  (graph, label).
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels).to(device_name)
