import numpy as np
import glob
import random
import h5py
import os
import re
import dgl
import torch
import scipy.sparse as sp

import math

from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from scipy.misc import imread
from scipy import ndimage

stl10_map={'airplane':0, 'bird':1,'car':2,'cat':3,'deer':4,'dog':5,'horse':6,'monkey':7,'ship':8,'truck':9}

cifar100_map={'couch': 25, 'pine_tree': 59, 'butterfly': 14, 'mountain': 49, 'bus': 13, 'sunflower': 82, 'beetle': 7, 'lamp': 40, 'can': 16, 'beaver': 4, 'bee': 6, 'elephant': 31, 'kangaroo': 38, 'oak_tree': 52, 'orchid': 54, 'trout': 91, 'pickup_truck': 58, 'chimpanzee': 21, 'worm': 99, 'snail': 77, 'television': 87, 'hamster': 36, 'squirrel': 80, 'lion': 43, 'otter': 55, 'shrew': 74, 'pear': 57, 'turtle': 93, 'seal': 72, 'plate': 61, 'fox': 34, 'train': 90, 'porcupine': 63, 'tulip': 92, 'keyboard': 39, 'orange': 53, 'possum': 64, 'cattle': 19, 'skyscraper': 76, 'bear': 3, 'cup': 28, 'cockroach': 24, 'caterpillar': 18, 'mouse': 50, 'forest': 33, 'rabbit': 65, 'aquarium_fish': 1, 'chair': 20, 'castle': 17, 'palm_tree': 56, 'telephone': 86, 'mushroom': 51, 'streetcar': 81, 'willow_tree': 96, 'man': 46, 'wardrobe': 94, 'bowl': 10, 'sweet_pepper': 83, 'maple_tree': 47, 'snake': 78, 'whale': 95, 'poppy': 62, 'tank': 85, 'bed': 5, 'rose': 70, 'crocodile': 27, 'raccoon': 66, 'tractor': 89, 'bicycle': 8, 'bridge': 12, 'dinosaur': 29, 'crab': 26, 'clock': 22, 'bottle': 9, 'lawn_mower': 41, 'road': 68, 'spider': 79, 'skunk': 75, 'tiger': 88, 'sea': 71, 'lizard': 44, 'cloud': 23, 'ray': 67, 'house': 37, 'lobster': 45, 'boy': 11, 'plain': 60, 'table': 84, 'dolphin': 30, 'camel': 15, 'rocket': 69, 'baby': 2, 'girl': 35, 'shark': 73, 'motorcycle': 48, 'flatfish': 32, 'leopard': 42, 'wolf': 97, 'apple': 0, 'woman': 98}

class ShockGraphDataset(Dataset):
    'Generates data for Keras'
    def __init__(self,directory,dataset,cache=True,symmetric=False,data_augment=False):
        'Initialization'
        
        self.directory = directory
        self.cache=cache
        self.symmetric = symmetric
        self.files=[]
        self.class_mapping={}
        self.sg_graphs=[]
        self.sg_labels=[]
        self.width=1
        self.center=np.zeros(2)
        self.sg_features=[]
        self.trans=()
        self.factor=1
        self.max_radius=1
        self.data_augment=data_augment
        
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
            features=self.sg_features[index]
        else:        
            item=self.files[index]
            graph,features=self.__read_shock_graph(item)
            obj=os.path.basename(item)
            obj=re.split(r'[0-9].*',obj)[0]
            class_name=obj[:obj.rfind('_')]
            label=self.class_mapping[class_name]

        if self.data_augment:
            self.__apply_da(graph,features)
        else:
            F_matrix=np.copy(features[0])
            self.__recenter(F_matrix)
            graph.ndata['h']=torch.from_numpy(F_matrix)
            
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
            graph,features=self.__read_shock_graph(fid)

            obj=os.path.basename(fid)
            obj=re.split(r'[0-9].*',obj)[0]
            class_name=obj[:obj.rfind('_')]
            label=self.class_mapping[class_name]

            self.sg_graphs.append(graph)
            self.sg_labels.append(label)
            self.sg_features.append(features)
            
    def __normalize_features(self,features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features

    def __round_nearest(self,x, a):
        return np.round(np.round(x / a) * a, -int(math.floor(math.log10(a))))

    def __translate_points(self,pt,v,length):
        i=0
        trans_points=np.zeros((pt.shape[0],6))
        for g in range(0,v.shape[1]):
            x=pt[:,0]+length*np.sin(v[:,g])
            y=pt[:,1]+length*np.cos(v[:,g])
            trans_points[:,i]=x
            i=i+1
            trans_points[:,i]=y
            i=i+1
            
        return trans_points

    def __get_pixel_values(self,F_matrix,color_space):

        pixel_values=np.zeros((F_matrix.shape[0],12),dtype=np.float32)

        zero_set=np.array([0.0,0.0])
        zcoord=np.array([0,1,2])
        for f in range(F_matrix.shape[0]):
            sg_xcoord=np.repeat(F_matrix[f,0],3)
            sg_ycoord=np.repeat(F_matrix[f,1],3)
            sg_cs=ndimage.map_coordinates(color_space,[sg_xcoord,sg_ycoord,zcoord])
            pixel_values[f,:3]=sg_cs
            
            
            bp1_xcoord=np.repeat(F_matrix[f,9],3)
            bp1_ycoord=np.repeat(F_matrix[f,10],3)
            bp1_cs=ndimage.map_coordinates(color_space,[bp1_xcoord,bp1_ycoord,zcoord])
            pixel_values[f,3:6]=bp1_cs
            
            if np.array_equal(F_matrix[f,11:13],zero_set)==False:
                bp2_xcoord=np.repeat(F_matrix[f,11],3)
                bp2_ycoord=np.repeat(F_matrix[f,12],3)
                bp2_cs=ndimage.map_coordinates(color_space,[bp2_xcoord,bp2_ycoord,zcoord])
                pixel_values[f,6:9]=bp2_cs
                
            if np.array_equal(F_matrix[f,13:15],zero_set)==False:
                bp3_xcoord=np.repeat(F_matrix[f,13],3)
                bp3_ycoord=np.repeat(F_matrix[f,14],3)
                bp3_cs=ndimage.map_coordinates(color_space,[bp3_xcoord,bp3_ycoord,zcoord])
                pixel_values[f,9:12]=bp3_cs

        return pixel_values/255.0
            
    def __apply_da(self,graph,features):

        F_matrix=np.copy(features[0])
        mask=features[1]
        
        flip=random.randint(0,1)
        if flip:
            F_matrix[:,1]=(self.width-F_matrix[:,1])-self.width/2
            F_matrix[:,3:6]=math.pi-F_matrix[:,3:6]
            F_matrix[:,15:18]=math.pi-F_matrix[:,15:18]
            
        random_scale=np.random.rand(1)*(2-.5)+.5
        random_scale=random_scale if random_scale != 0 else 1
        random_trans=np.random.rand(2)*10-5
        
        F_matrix[:,:2]+=random_trans
        F_matrix[:,:2]*=random_scale
        F_matrix[:,2]*=random_scale

        v=F_matrix[:,3:6]+F_matrix[:,6:9]
        new_trans_points=self.__translate_points(F_matrix[:,:2],v,F_matrix[:,2])
        new_trans_points=new_trans_points*mask

        F_matrix[:,9:15]=new_trans_points

        self.trans=(flip,random_scale,random_trans)
        self.__recenter(F_matrix)

        graph.ndata['h']=torch.from_numpy(F_matrix)

    def __recenter(self,F_matrix):

        # radius of shock point
        F_matrix[:,2] /= self.max_radius*2.0

        # theta of node
        F_matrix[:,3] /= 2.0*math.pi
        F_matrix[:,4] /= 2.0*math.pi
        F_matrix[:,5] /= 2.0*math.pi

        # phi of node
        F_matrix[:,6] /= math.pi
        F_matrix[:,7] /= math.pi
        F_matrix[:,8] /= math.pi

        # plus theta
        F_matrix[:,15] /= 2.0*math.pi
        F_matrix[:,16] /= 2.0*math.pi
        F_matrix[:,17] /= 2.0*math.pi

        # remove ref pt for contour and shock points
        F_matrix[:,:2] -=self.center

        zero_set=np.array([0.0,0.0])

        for row_idx in range(0,F_matrix.shape[0]):
            F_matrix[row_idx,9:11]-=self.center
            
            if np.array_equal(F_matrix[row_idx,11:13],zero_set)==False:
                F_matrix[row_idx,11:13]-=self.center
                
            if np.array_equal(F_matrix[row_idx,13:15],zero_set)==False:
                F_matrix[row_idx,13:15]-=self.center

        F_matrix[:,:2] /= self.factor
        F_matrix[:,9:11] /= self.factor
        F_matrix[:,11:13] /= self.factor
        F_matrix[:,13:15] /= self.factor
                            
    def __unwrap_data(self,F_matrix,debug_matrix):

        #make a copy
        feature_matrix=F_matrix

        # get debug data
        ref_pt=debug_matrix[:2]
        max_offsets=debug_matrix[2:4]
        max_radius=debug_matrix[4]

        self.max_radius=max_radius
        
        # shock pt location
        feature_matrix[:,:2] *= max_offsets

        # radius of shock point
        feature_matrix[:,2] *= max_radius

        # theta of node
        feature_matrix[:,3] *= 2.0*math.pi
        feature_matrix[:,4] *= 2.0*math.pi
        feature_matrix[:,5] *= 2.0*math.pi

        # phi of node
        feature_matrix[:,6] *= math.pi
        feature_matrix[:,7] *= math.pi
        feature_matrix[:,8] *= math.pi

        # left boundary point
        feature_matrix[:,9:11] *= max_offsets
        feature_matrix[:,11:13] *= max_offsets
        feature_matrix[:,13:15] *= max_offsets

        # plus theta
        feature_matrix[:,15] *= math.pi
        feature_matrix[:,16] *= math.pi
        feature_matrix[:,17] *= math.pi

        # remove ref pt for contour and shock points
        feature_matrix[:,0] +=ref_pt[0]
        feature_matrix[:,1] +=ref_pt[1]

        zero_set=np.array([0.0,0.0])

        mask=np.ones((feature_matrix.shape[0],6))
        for row_idx in range(0,feature_matrix.shape[0]):
            feature_matrix[row_idx,9:11]+=ref_pt
            
            if np.array_equal(feature_matrix[row_idx,11:13],zero_set)==False:
                feature_matrix[row_idx,11:13]+=ref_pt
            else:
                mask[row_idx,2:4]=0
                
            if np.array_equal(feature_matrix[row_idx,13:15],zero_set)==False:
                feature_matrix[row_idx,13:15]+=ref_pt
            else:
                mask[row_idx,4:6]=0

        return feature_matrix,mask
        
    def __read_shock_graph(self,sg_file):
        fid=h5py.File(sg_file,'r')

        # read in debug info
        debug_data=fid.get('debug')
        debug_matrix=np.array(debug_data)

        # read in features
        feature_data=fid.get('feature')
        F_matrix=np.array(feature_data).astype(np.float32)
        
        # read in adj matrix
        adj_data=fid.get('adj_matrix')
        adj_matrix=np.array(adj_data)

        # remove normalization put in
        F_matrix_unwrapped,mask=self.__unwrap_data(F_matrix,debug_matrix)
        
        # center of bounding box
        # will be a constant across
        self.width=(F_matrix_unwrapped[1,1]-F_matrix_unwrapped[0,1])
        self.center=np.array([F_matrix_unwrapped[1,1]-self.width/2.0,
                              F_matrix_unwrapped[1,1]-self.width/2.0])
        self.factor=self.width*1.5

        # get image
        image_name=sg_file.split('-')[0]+'.png'
        img=imread(image_name)
        F_color=self.__get_pixel_values(F_matrix_unwrapped,img)

        # add color and shape features together
        F_combined=np.concatenate((F_matrix_unwrapped,F_color),axis=1)

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
                                
        return G,(F_combined,mask)

def collate(samples,device_name):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels).to(device_name)
