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
from operator import itemgetter
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize

stl10_map={'airplane':0, 'bird':1,'car':2,'cat':3,'deer':4,'dog':5,'horse':6,'monkey':7,'ship':8,'truck':9}

cifar100_map={'couch': 25, 'pine_tree': 59, 'butterfly': 14, 'mountain': 49, 'bus': 13, 'sunflower': 82, 'beetle': 7, 'lamp': 40, 'can': 16, 'beaver': 4, 'bee': 6, 'elephant': 31, 'kangaroo': 38, 'oak_tree': 52, 'orchid': 54, 'trout': 91, 'pickup_truck': 58, 'chimpanzee': 21, 'worm': 99, 'snail': 77, 'television': 87, 'hamster': 36, 'squirrel': 80, 'lion': 43, 'otter': 55, 'shrew': 74, 'pear': 57, 'turtle': 93, 'seal': 72, 'plate': 61, 'fox': 34, 'train': 90, 'porcupine': 63, 'tulip': 92, 'keyboard': 39, 'orange': 53, 'possum': 64, 'cattle': 19, 'skyscraper': 76, 'bear': 3, 'cup': 28, 'cockroach': 24, 'caterpillar': 18, 'mouse': 50, 'forest': 33, 'rabbit': 65, 'aquarium_fish': 1, 'chair': 20, 'castle': 17, 'palm_tree': 56, 'telephone': 86, 'mushroom': 51, 'streetcar': 81, 'willow_tree': 96, 'man': 46, 'wardrobe': 94, 'bowl': 10, 'sweet_pepper': 83, 'maple_tree': 47, 'snake': 78, 'whale': 95, 'poppy': 62, 'tank': 85, 'bed': 5, 'rose': 70, 'crocodile': 27, 'raccoon': 66, 'tractor': 89, 'bicycle': 8, 'bridge': 12, 'dinosaur': 29, 'crab': 26, 'clock': 22, 'bottle': 9, 'lawn_mower': 41, 'road': 68, 'spider': 79, 'skunk': 75, 'tiger': 88, 'sea': 71, 'lizard': 44, 'cloud': 23, 'ray': 67, 'house': 37, 'lobster': 45, 'boy': 11, 'plain': 60, 'table': 84, 'dolphin': 30, 'camel': 15, 'rocket': 69, 'baby': 2, 'girl': 35, 'shark': 73, 'motorcycle': 48, 'flatfish': 32, 'leopard': 42, 'wolf': 97, 'apple': 0, 'woman': 98}

imageclef_map={'aeroplane':0, 'bike':1, 'bird':2, 'boat':3, 'bottle':4, 'bus':5, 'car':6, 'dog':7, 'horse':8, 'monitor':9, 'motorbike':10, 'people':11}

office31_map={'back_pack':0,
              'bike':1,
              'bike_helmet':2,
              'bookcase':3,
              'bottle':4,
              'calculator':5,
              'desk_chair':6,
              'desk_lamp':7,
              'desktop_computer':8,
              'file_cabinet':9,
              'headphones':10,
              'keyboard':11,
              'laptop_computer':12,
              'letter_tray':13,
              'mobile_phone':14,
              'monitor':15,
              'mouse':16,
              'mug':17,
              'paper_notebook':18,
              'pen':19,
              'phone':20,
              'printer':21,
              'projector':22,
              'punchers':23,
              'ring_binder':24,
              'ruler':25,
              'scissors':26,
              'speaker':27,
              'stapler':28,
              'tape_dispenser':29,
              'trash_can':30}



def fixAngleMPiPi_new_vector(vec):
    output=np.zeros(np.shape(vec))
    it=np.nditer(vec,flags=['f_index'])
    for a in it:
        if a < -math.pi:
            output[it.index]=a+2.0*math.pi
        elif a > math.pi:
            output[it.index]=a-2.0*math.pi;
        else:
            output[it.index]=a

    return output

def fixAngle2PiPi_new_vector(vec):
    output=np.zeros(np.shape(vec))
    it=np.nditer(vec,flags=['f_index'])
    for a in it:
        if a < -2.0*math.pi:
            output[it.index]=a+2.0*math.pi
        elif a > 2.0*math.pi:
            output[it.index]=a-2.0*math.pi;
        else:
            output[it.index]=a

    return output

class ShockGraphDataset(Dataset):
    'Generates data for Keras'
    def __init__(self,directory,dataset,app=False,cache=True,symmetric=False,data_augment=False,flip_pp=False,grid=8,self_loop=False,dsm_norm=True):
        'Initialization'
        
        self.directory = directory
        self.app=app
        self.cache=cache
        self.symmetric = symmetric
        self.files=[]
        self.class_mapping={}
        self.sg_graphs=[]
        self.sg_labels=[]
        self.adj_matrices=[]
        self.width=1
        self.center=np.zeros(2)
        self.image_size=1
        self.sg_features=[]
        self.trans=()
        self.factor=1
        self.max_radius=1
        self.flip_pp=flip_pp
        self.data_augment=data_augment
        self.grid=grid
        self.grid_mapping=[]
        self.self_loop=self_loop
        self.dsm_norm=dsm_norm
        
        if dataset=='cifar100':
            print('Using cifar 100 dataset')
            self.class_mapping=cifar100_map
        elif dataset=='stl10':
            print('Using stl 10 dataset')
            self.class_mapping=stl10_map
        elif dataset=='imageclef':
            print('Using image-clef dataset')
            self.class_mapping=imageclef_map
        else:
            print('Using office 31 dataset')
            self.class_mapping=office31_map
            
        self.__gen_file_list()
        self.__preprocess_graphs()
            
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.files)

    def __getitem__(self, index):
        'Generate one example of data'
        
        features=self.sg_features[index]
        label=self.sg_labels[index]
        spp_map=self.grid_mapping[index]

        if self.cache:
            graph=self.sg_graphs[index]
            F_matrix=np.copy(features[0])
            F_matrix=self.__recenter(F_matrix,absolute=True)
            graph.ndata['h']=torch.from_numpy(F_matrix)
        else:        
            adj_matrix=self.adj_matrices[index]        

            if self.data_augment:
                new_adj,new_F,spp_map=self.__apply_da(adj_matrix,features)
            else:
                new_adj=adj_matrix
                new_F=features[0]
                spp_map=self.__compute_spp_map(new_F,self.grid)
                self.__recenter(new_F,absolute=True)

            graph=self.__create_graph(new_adj)
            graph.ndata['h']=torch.from_numpy(new_F)

        graph.ndata['x']=torch.from_numpy(spp_map)
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

    def __dsm_normalize(self,M):

        if self.dsm_norm==False:
            return M

        if self.symmetric:
            M=M+M.transpose()

        if self.self_loop:
            M = M + np.eye(M.shape[0])
                    
        M=normalize(M,norm='l1',axis=1)
        M=normalize(M,norm='l2',axis=0)
        return M.astype(np.float32)
    
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
        self.files.sort()
        
    def __compute_spp_map(self, F_matrix,cells):
        grid=np.linspace(0,self.image_size,cells+1)
        grid_map=defaultdict(list)
        
        for idx in range(F_matrix.shape[0]):
            pts=F_matrix[idx,:]
            xloc=max(np.searchsorted(grid,pts[0])-1,0,0)
            yloc=max(np.searchsorted(grid,pts[1])-1,0,0)
            grid_map[(xloc,yloc)].append(idx)

        grid_cell=np.ones((F_matrix.shape[0],250),dtype=np.int32)*-1

        idx=0
        for key,value in grid_map.items():
            grid_cell[idx,0]=key[0]
            grid_cell[idx,1]=key[1]
            grid_cell[idx,2]=len(value)
            
            col=3
            for ss in value:
                grid_cell[idx,col]=ss
                col+=1

            idx+=1

        return grid_cell

    def __preprocess_graphs(self):

        for fid in tqdm(self.files):
            adj_matrix,features,edge_features=self.__read_shock_graph(fid)

            obj=os.path.basename(fid)
            obj=re.split(r'[0-9].*',obj)[0]
            class_name=obj[:obj.rfind('_')]
            label=self.class_mapping[class_name]
            grid_cell=self.__compute_spp_map(features[0],self.grid)
            
            self.adj_matrices.append(adj_matrix)
            self.sg_labels.append(label)
            self.sg_features.append(features)
            self.grid_mapping.append(grid_cell)
            
            if self.cache:
                graph=self.__create_graph(adj_matrix,edge_features)
                self.sg_graphs.append(graph)

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

        pixel_values=np.zeros((F_matrix.shape[0],21),dtype=np.float32)

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

            bp1_xcoord=np.repeat(F_matrix[f,15],3)
            bp1_ycoord=np.repeat(F_matrix[f,16],3)
            bp1_cs=ndimage.map_coordinates(color_space,[bp1_xcoord,bp1_ycoord,zcoord])
            pixel_values[f,12:15]=bp1_cs

            if np.array_equal(F_matrix[f,11:13],zero_set)==False:
                bp2_xcoord=np.repeat(F_matrix[f,11],3)
                bp2_ycoord=np.repeat(F_matrix[f,12],3)
                bp2_cs=ndimage.map_coordinates(color_space,[bp2_xcoord,bp2_ycoord,zcoord])
                pixel_values[f,6:9]=bp2_cs

                bp2_xcoord=np.repeat(F_matrix[f,17],3)
                bp2_ycoord=np.repeat(F_matrix[f,18],3)
                bp2_cs=ndimage.map_coordinates(color_space,[bp2_xcoord,bp2_ycoord,zcoord])
                pixel_values[f,15:18]=bp2_cs

            if np.array_equal(F_matrix[f,13:15],zero_set)==False:
                bp3_xcoord=np.repeat(F_matrix[f,13],3)
                bp3_ycoord=np.repeat(F_matrix[f,14],3)
                bp3_cs=ndimage.map_coordinates(color_space,[bp3_xcoord,bp3_ycoord,zcoord])
                pixel_values[f,9:12]=bp3_cs

                bp3_xcoord=np.repeat(F_matrix[f,19],3)
                bp3_ycoord=np.repeat(F_matrix[f,20],3)
                bp3_cs=ndimage.map_coordinates(color_space,[bp3_xcoord,bp3_ycoord,zcoord])
                pixel_values[f,18:21]=bp3_cs

        return pixel_values/255.0
            
    def __apply_da(self,orig_adj_matrix,features):

        F_matrix=np.copy(features[0])
        mask=np.copy(features[1])

        flip=random.randint(0,1)
        if flip:
            F_matrix[:,1]=((self.width-1)-F_matrix[:,1])-self.width/2
            F_matrix[:,3:6]=math.pi-F_matrix[:,3:6]
            
        random_scale=np.random.rand(1)*(2-.5)+.5
        random_scale=random_scale if random_scale != 0 else 1
        random_trans=np.random.randint(-20,20,size=2)
        
        F_matrix[:,:2]+=random_trans
        F_matrix[:,:2]*=random_scale
        F_matrix[:,2]*=random_scale

        v_plus=F_matrix[:,3:6]+F_matrix[:,6:9]
        new_plus=self.__translate_points(F_matrix[:,:2],v_plus,F_matrix[:,2])
        v_minus=F_matrix[:,3:6]-F_matrix[:,6:9]
        new_minus=self.__translate_points(F_matrix[:,:2],v_minus,F_matrix[:,2])
                                             
        F_matrix[:,9:15]=new_plus
        F_matrix[:,15:21]=new_minus

        if flip:
            F_matrix[:,3:6]+=2*math.pi
            F_matrix[:,3]=fixAngle2PiPi_new_vector(F_matrix[:,3])
            F_matrix[:,4]=fixAngle2PiPi_new_vector(F_matrix[:,4])
            F_matrix[:,5]=fixAngle2PiPi_new_vector(F_matrix[:,5])
        
            # plus theta
            F_matrix[:,21] = fixAngleMPiPi_new_vector(F_matrix[:,3]+F_matrix[:,6]-(math.pi/2.0));
            F_matrix[:,22] = fixAngleMPiPi_new_vector(F_matrix[:,4]+F_matrix[:,7]-(math.pi/2.0));
            F_matrix[:,23] = fixAngleMPiPi_new_vector(F_matrix[:,5]+F_matrix[:,8]-(math.pi/2.0));
        
            # minus theta
            F_matrix[:,24] = fixAngleMPiPi_new_vector(F_matrix[:,3]-F_matrix[:,6]+math.pi/2.0);
            F_matrix[:,25] = fixAngleMPiPi_new_vector(F_matrix[:,4]-F_matrix[:,7]+math.pi/2.0);
            F_matrix[:,26] = fixAngleMPiPi_new_vector(F_matrix[:,5]-F_matrix[:,8]+math.pi/2.0);

        #scale arclength, curvature
        F_matrix[:,28:58] *= random_scale
        
        #mask
        F_matrix=F_matrix*mask
        
        self.trans=(flip,random_scale[0],random_trans)
        F_pruned,adj_pruned,mask_pruned=self.__prune_ob(F_matrix,orig_adj_matrix,mask,self.image_size*random_scale)
        new_adj_matrix,new_F_matrix,new_mask=self.__compute_sorted_order(F_pruned,adj_pruned,mask_pruned)

        new_center=np.zeros(2)
        new_center[0]=(self.image_size*random_scale)/2.0
        new_center[1]=(self.image_size*random_scale)/2.0
        factor=((self.image_size*random_scale)/2.0)*1.2

        spp_map=self.__compute_spp_map(new_F_matrix,self.grid)

        self.__recenter(new_F_matrix,new_center,factor,absolute=True)
        
        return new_adj_matrix,new_F_matrix,spp_map

    def __compute_sorted_order(self,F_matrix,orig_adj_matrix,mask):

        node_order={}
        
        # define a sorting order for nodes in graph
        key_tuples=[]
        for i in range(F_matrix.shape[0]):
            key_tuples.append((F_matrix[i][0],
                               F_matrix[i][1],
                               F_matrix[i][2],
                               i))
            
        sorted_tuples=sorted(key_tuples,key=itemgetter(0,1,2),reverse=False)
        new_adj_matrix=np.zeros((orig_adj_matrix.shape[0],orig_adj_matrix.shape[0]))
        
        new_list=[]
        for idx in range(0,len(sorted_tuples)):
            key=sorted_tuples[idx][3]
            value=idx
            node_order[key]=value
            new_list.append(key)
            
 
        for row in range(0,orig_adj_matrix.shape[0]):
            source_idx=node_order[row]
            neighbors=orig_adj_matrix[row,:]
            target=np.nonzero(neighbors)[0]
            for t in target:
                target_idx=node_order[t]
                new_adj_matrix[source_idx][target_idx]=1

        new_F_matrix=F_matrix[new_list,:]
        new_mask=mask[new_list,:]
        return new_adj_matrix,new_F_matrix,new_mask
    
    def __recenter(self,F_matrix,center=None,factor=None,absolute=True):

        if center is None:
            center=self.center

        if factor is None:
            factor=self.factor
            
        # radius of shock point
        if absolute:

            rad_scale=68.2340
            angle_scale=713.8326
            length_scale=116.8454
            curve_scale=10.817794298253000
            poly_scale=8272.887700675255
 
            # scale shock radius
            F_matrix[:,2] /= rad_scale

            # scale shock curvature
            F_matrix[:,28] /= curve_scale
            F_matrix[:,38] /= curve_scale
            F_matrix[:,48] /= curve_scale

            # scale plus curvature
            F_matrix[:,31] /= curve_scale
            F_matrix[:,41] /= curve_scale
            F_matrix[:,51] /= curve_scale

            # scale minus curvature
            F_matrix[:,34] /= curve_scale
            F_matrix[:,44] /= curve_scale
            F_matrix[:,54] /= curve_scale

            # scale shock length
            F_matrix[:,29] /= length_scale
            F_matrix[:,39] /= length_scale
            F_matrix[:,49] /= length_scale

            # scale plus length
            F_matrix[:,32] /= length_scale
            F_matrix[:,42] /= length_scale
            F_matrix[:,52] /= length_scale

            # scale minus length
            F_matrix[:,35] /= length_scale
            F_matrix[:,45] /= length_scale
            F_matrix[:,55] /= length_scale

            # scale shock angle
            F_matrix[:,30] /= angle_scale
            F_matrix[:,40] /= angle_scale
            F_matrix[:,50] /= angle_scale

            # scale plus angle
            F_matrix[:,33] /= angle_scale
            F_matrix[:,43] /= angle_scale
            F_matrix[:,53] /= angle_scale

            # scale minus angle
            F_matrix[:,36] /= angle_scale
            F_matrix[:,46] /= angle_scale
            F_matrix[:,56] /= angle_scale

            # scale poly scale
            F_matrix[:,37] /= poly_scale
            F_matrix[:,47] /= poly_scale
            F_matrix[:,57] /= poly_scale

        else:

            # scale radius
            rad_scale=np.max(F_matrix[:,2])
            F_matrix[:,2] /= rad_scale

            curve_scale=np.max(np.abs(np.concatenate((F_matrix[:,28],F_matrix[:,38],F_matrix[:,48],
                                                      F_matrix[:,31],F_matrix[:,41],F_matrix[:,51],
                                                      F_matrix[:,34],F_matrix[:,44],F_matrix[:,54]),axis=0)))

            # scale shock curvature
            F_matrix[:,28] /= curve_scale
            F_matrix[:,38] /= curve_scale
            F_matrix[:,48] /= curve_scale

            # scale plus curvature
            F_matrix[:,31] /= curve_scale
            F_matrix[:,41] /= curve_scale
            F_matrix[:,51] /= curve_scale

            # scale minus curvature
            F_matrix[:,34] /= curve_scale
            F_matrix[:,44] /= curve_scale
            F_matrix[:,54] /= curve_scale

            length_scale=np.max(np.abs(np.concatenate((F_matrix[:,29],F_matrix[:,39],F_matrix[:,49],
                                                       F_matrix[:,32],F_matrix[:,42],F_matrix[:,52],
                                                       F_matrix[:,35],F_matrix[:,45],F_matrix[:,55]),axis=0)))
            
            # scale shock length
            F_matrix[:,29] /= length_scale
            F_matrix[:,39] /= length_scale
            F_matrix[:,49] /= length_scale

            # scale plus length
            F_matrix[:,32] /= length_scale
            F_matrix[:,42] /= length_scale
            F_matrix[:,52] /= length_scale

            #scale minus length
            F_matrix[:,35] /= length_scale
            F_matrix[:,45] /= length_scale
            F_matrix[:,55] /= length_scale

            angle_scale=np.max(np.abs(np.concatenate((F_matrix[:,30],F_matrix[:,40],F_matrix[:,50],
                                                      F_matrix[:,33],F_matrix[:,43],F_matrix[:,53],
                                                      F_matrix[:,36],F_matrix[:,46],F_matrix[:,56]),axis=0)))


            # scale shock angle
            F_matrix[:,30] /= angle_scale
            F_matrix[:,40] /= angle_scale
            F_matrix[:,50] /= angle_scale

            # scale plus angle
            F_matrix[:,33] /= angle_scale
            F_matrix[:,43] /= angle_scale
            F_matrix[:,53] /= angle_scale

            #scale minus angle
            F_matrix[:,36] /= angle_scale
            F_matrix[:,46] /= angle_scale
            F_matrix[:,56] /= angle_scale

            poly_scale=np.max(np.abs(np.concatenate((F_matrix[:,37],F_matrix[:,47],F_matrix[:,57]),axis=0)))

            F_matrix[:,37] /= poly_scale
            F_matrix[:,47] /= poly_scale
            F_matrix[:,57] /= poly_scale

        # theta of node
        F_matrix[:,3] /= 2.0*math.pi
        F_matrix[:,4] /= 2.0*math.pi
        F_matrix[:,5] /= 2.0*math.pi

        # phi of node
        F_matrix[:,6] /= math.pi
        F_matrix[:,7] /= math.pi
        F_matrix[:,8] /= math.pi

        # plus theta
        F_matrix[:,21] /= math.pi
        F_matrix[:,22] /= math.pi
        F_matrix[:,23] /= math.pi

        # minus theta
        F_matrix[:,24] /= math.pi
        F_matrix[:,25] /= math.pi
        F_matrix[:,26] /= math.pi

        # remove ref pt for contour and shock points
        F_matrix[:,:2] -=center

        zero_set=np.array([0.0,0.0])

        for row_idx in range(0,F_matrix.shape[0]):
            F_matrix[row_idx,9:11]-=center
            F_matrix[row_idx,15:17]-=center
            
            if np.array_equal(F_matrix[row_idx,11:13],zero_set)==False:
                F_matrix[row_idx,11:13]-=center
                F_matrix[row_idx,17:19]-=center
                
            if np.array_equal(F_matrix[row_idx,13:15],zero_set)==False:
                F_matrix[row_idx,13:15]-=center
                F_matrix[row_idx,19:21]-=center

        F_matrix[:,:2] /= factor
        F_matrix[:,9:15] /= factor
        F_matrix[:,15:21] /= factor

        F_matrix=np.delete(F_matrix,np.s_[28:F_matrix.shape[1]],1)
        return F_matrix
        

    def __prune_ob(self,F_matrix,adj_matrix,mask,scale):

        # first round of delete
        rows=np.unique(np.where((F_matrix[:,:2]<0)|(F_matrix[:,:2]>=scale))[0])
        F_matrix=np.delete(F_matrix,rows,axis=0)
        adj_matrix=np.delete(adj_matrix,rows,axis=0)
        adj_matrix=np.delete(adj_matrix,rows,axis=1)
        mask=np.delete(mask,rows,axis=0)

        # check if any disconnected nodes
        out_degree=np.where(~adj_matrix.any(axis=1))[0]
        in_degree=np.where(~adj_matrix.any(axis=0))[0]
        rows=np.intersect1d(out_degree,in_degree)

        # second round of delete
        if rows.shape[0]:
            F_matrix=np.delete(F_matrix,rows,axis=0)
            adj_matrix=np.delete(adj_matrix,rows,axis=0)
            adj_matrix=np.delete(adj_matrix,rows,axis=1)
            mask=np.delete(mask,rows,axis=0)

        return F_matrix,adj_matrix,mask
        
    def __unwrap_data(self,F_matrix):

        #make a copy
        feature_matrix=F_matrix
        
        # theta of node
        feature_matrix[:,3] *= 2.0*math.pi
        feature_matrix[:,4] *= 2.0*math.pi
        feature_matrix[:,5] *= 2.0*math.pi

        # phi of node
        feature_matrix[:,6] *= math.pi
        feature_matrix[:,7] *= math.pi
        feature_matrix[:,8] *= math.pi

        # plus theta
        feature_matrix[:,21] *= math.pi
        feature_matrix[:,22] *= math.pi
        feature_matrix[:,23] *= math.pi

        # minus theta
        feature_matrix[:,24] *= math.pi
        feature_matrix[:,25] *= math.pi
        feature_matrix[:,26] *= math.pi

        zero_set=np.array([0.0,0.0])

        mask=np.ones(feature_matrix.shape,dtype=np.float32)
        for row_idx in range(0,feature_matrix.shape[0]):
                        
            if np.array_equal(feature_matrix[row_idx,11:13],zero_set)==True:
                mask[row_idx,11:13]=0
                mask[row_idx,4]=0
                mask[row_idx,7]=0
                mask[row_idx,22]=0
                mask[row_idx,25]=0
                mask[row_idx,17:19]=0

            if np.array_equal(feature_matrix[row_idx,13:15],zero_set)==True:
                mask[row_idx,13:15]=0
                mask[row_idx,5]=0
                mask[row_idx,8]=0
                mask[row_idx,23]=0
                mask[row_idx,26]=0
                mask[row_idx,19:21]=0


        return feature_matrix,mask
        
    def __read_shock_graph(self,sg_file):
        fid=h5py.File(sg_file,'r')

        # read in features
        feature_data=fid.get('feature')
        F_matrix=np.array(feature_data).astype(np.float32)
        
        # read in adj matrix
        adj_data=fid.get('adj_matrix')
        adj_matrix=np.array(adj_data)

        edge_features=[]

        rad_scale=68.2340
        angle_scale=713.8326
        length_scale=116.8454
        curve_scale=10.817794298253000
        poly_scale=8272.887700675255


        # read in edge channel data
        ec0_data=fid.get('edge_chan_0')
        ec0=np.array(ec0_data)
        ec0=(ec0/curve_scale).astype(np.float32)
        ec0=self.__dsm_normalize(ec0)

        # read in edge channel data
        ec1_data=fid.get('edge_chan_1')
        ec1=np.array(ec1_data)
        ec1=(ec1/length_scale).astype(np.float32)
        ec1=self.__dsm_normalize(ec1)
        
        # read in edge channel data
        ec2_data=fid.get('edge_chan_2')
        ec2=np.array(ec2_data)
        ec2=(ec2/angle_scale).astype(np.float32)
        ec2=self.__dsm_normalize(ec2)

        # read in edge channel data
        ec3_data=fid.get('edge_chan_3')
        ec3=np.array(ec3_data)
        ec3=(ec3/curve_scale).astype(np.float32)
        ec3=self.__dsm_normalize(ec3)
        
        # read in edge channel data
        ec4_data=fid.get('edge_chan_4')
        ec4=np.array(ec4_data)
        ec4=(ec4/length_scale).astype(np.float32)
        ec4=self.__dsm_normalize(ec4)

        # read in edge channel data
        ec5_data=fid.get('edge_chan_5')
        ec5=np.array(ec5_data)
        ec5=(ec5/angle_scale).astype(np.float32)
        ec5=self.__dsm_normalize(ec5)

        # read in edge channel data
        ec6_data=fid.get('edge_chan_6')
        ec6=np.array(ec6_data)
        ec6=(ec6/curve_scale).astype(np.float32)
        ec6=self.__dsm_normalize(ec6)
        
        # read in edge channel data
        ec7_data=fid.get('edge_chan_7')
        ec7=np.array(ec7_data)
        ec7=(ec7/length_scale).astype(np.float32)
        ec7=self.__dsm_normalize(ec7)

        # read in edge channel data
        ec8_data=fid.get('edge_chan_8')
        ec8=np.array(ec8_data)
        ec8=(ec8/angle_scale).astype(np.float32)
        ec8=self.__dsm_normalize(ec8)
        
        # read in edge channel data
        ec9_data=fid.get('edge_chan_9')
        ec9=np.array(ec9_data)
        ec9=(ec9/poly_scale).astype(np.float32)
        ec9=self.__dsm_normalize(ec9)

        edge_features=torch.from_numpy(np.dstack((ec0,ec1,ec2,ec3,ec4,ec5,ec6,ec7,ec8,ec9)))

        # read in image dims
        dims=fid.get('dims')
        dims=np.array(dims)
        
        # remove normalization put in
        F_matrix_unwrapped,mask=self.__unwrap_data(F_matrix)
        
        # center of bounding box
        # will be a constant across
        self.width=(F_matrix_unwrapped[1,1]-F_matrix_unwrapped[0,1])
        self.image_size=dims[0]
        self.center=np.array([self.image_size/2.0,self.image_size/2.0])
        self.factor=(self.image_size/2.0)*1.2

        # get image
        if self.app:
            image_name=sg_file.split('-')[0]+'.png'
            img=imread(image_name)
            F_color=self.__get_pixel_values(F_matrix_unwrapped,img)

            # add color and shape features together
            F_combined=np.concatenate((F_matrix_unwrapped,F_color),axis=1)
        else:
            F_combined=F_matrix_unwrapped

        # F_combined_pruned,adj_matrix_pruned,mask_pruned=self.__prune_ob(F_combined,adj_matrix,mask,self.image_size)
            
        # if self.flip_pp:
        #     F_combined_pruned[:,1]=((self.width-1)-F_combined_pruned[:,1])-self.width/2
        #     F_combined_pruned[:,3:6]=math.pi-F_combined_pruned[:,3:6]

        #     v_plus=F_combined_pruned[:,3:6]+F_combined_pruned[:,6:9]
        #     new_plus=self.__translate_points(F_combined_pruned[:,:2],v_plus,F_combined_pruned[:,2])
        #     v_minus=F_combined_pruned[:,3:6]-F_combined_pruned[:,6:9]
        #     new_minus=self.__translate_points(F_combined_pruned[:,:2],v_minus,F_combined_pruned[:,2])

        #     F_combined_pruned[:,9:15]=new_plus
        #     F_combined_pruned[:,15:21]=new_minus

        #     F_combined_pruned[:,3:6]+=2*math.pi
        #     F_combined_pruned[:,3]=fixAngle2PiPi_new_vector(F_combined_pruned[:,3])
        #     F_combined_pruned[:,4]=fixAngle2PiPi_new_vector(F_combined_pruned[:,4])
        #     F_combined_pruned[:,5]=fixAngle2PiPi_new_vector(F_combined_pruned[:,5])

        #     # plus theta
        #     F_combined_pruned[:,21] = fixAngleMPiPi_new_vector(F_combined_pruned[:,3]+F_combined_pruned[:,6]-(math.pi/2.0));
        #     F_combined_pruned[:,22] = fixAngleMPiPi_new_vector(F_combined_pruned[:,4]+F_combined_pruned[:,7]-(math.pi/2.0));
        #     F_combined_pruned[:,23] = fixAngleMPiPi_new_vector(F_combined_pruned[:,5]+F_combined_pruned[:,8]-(math.pi/2.0));

        #     # minus theta
        #     F_combined_pruned[:,24] = fixAngleMPiPi_new_vector(F_combined_pruned[:,3]-F_combined_pruned[:,6]+math.pi/2.0);
        #     F_combined_pruned[:,25] = fixAngleMPiPi_new_vector(F_combined_pruned[:,4]-F_combined_pruned[:,7]+math.pi/2.0);
        #     F_combined_pruned[:,26] = fixAngleMPiPi_new_vector(F_combined_pruned[:,5]-F_combined_pruned[:,8]+math.pi/2.0);

        #     F_combined_pruned=F_combined_pruned*mask_pruned

        # # resort to be safe
        # new_adj_matrix,new_F_matrix,new_mask=self.__compute_sorted_order(F_combined_pruned,adj_matrix_pruned,mask_pruned)
            
        return adj_matrix,(F_combined,mask),edge_features


    def __create_graph(self,adj_matrix,edge_features=[]):

        # convert to dgl
        G=dgl.DGLGraph()
        G.add_nodes(adj_matrix.shape[0])
        G.set_n_initializer(dgl.init.zero_initializer)
 
        for row in range(adj_matrix.shape[0]):
            neighbors=adj_matrix[row,:]
            target=np.nonzero(neighbors)[0]
            source=np.zeros(target.shape)+row

            if self.self_loop:
                G.add_edges(row,row)
                temp=edge_features[np.int32(row),np.int32(row),:]
                edata=np.reshape(temp,(1,temp.shape[0]))
                G.edges[row,row].data['e']=edata
                
            if target.size:
                G.add_edges(source,target)
                edata=edge_features[np.int32(source),np.int32(target),:]
                G.edges[source,target].data['e']=edata

                if self.symmetric:
                    for idx in range(len(target)):
                        if adj_matrix[np.int32(target[idx]),np.int32(source[idx])]==0:
                            G.add_edges(target[idx],source[idx])
                            temp=edge_features[np.int32(target[idx]),np.int32(source[idx]),:]
                            edata=np.reshape(temp,(1,temp.shape[0]))
                            G.edges[target[idx],source[idx]].data['e']=edata
                        else:
                            print(target[idx],source[idx],'revisited')
                    
        return G
     
def collate(samples,device_name):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels).to(device_name)
