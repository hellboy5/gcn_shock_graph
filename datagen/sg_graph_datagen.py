import numpy as np
import keras 
import glob
import random
import h5py
import os
import re

from scipy.sparse import coo_matrix,csr_matrix

class ShockGraphDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,directory, labels, numb_nodes,numb_attrs,numb_filters,batch_size=32,
                 n_classes=10, symmetric=False,sparse=False,shuffle=True):
        'Initialization'
        
        self.directory = directory
        self.labels = labels
        self.numb_nodes = numb_nodes
        self.numb_attrs = numb_attrs
        self.numb_filters = numb_filters
        self.batch_size=batch_size
        self.n_classes = n_classes
        self.symmetric = symmetric
        self.sparse = sparse
        self.shuffle = shuffle
        self.files=[]
        self.class_mapping={}
        
        self.__gen_file_list()
        self.__parse_label()
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        samples=np.arange(index*self.batch_size,(index+1)*self.batch_size)
        batch_indices = np.remainder(samples,len(self.files))

        class_id=np.zeros(self.batch_size,)
        adj_batch=np.zeros((self.batch_size,self.numb_nodes*self.numb_filters,self.numb_nodes))
        feature_batch=np.zeros((self.batch_size,self.numb_nodes,self.numb_attrs))

        for idx in range(0,len(batch_indices)):
            item=self.files[batch_indices[idx]]

            if self.sparse:
                adj_mat,feature_mat=self.__read_shock_graph_sparse(item)
            else:
                adj_mat,feature_mat=self.__read_shock_graph(item)
                
            if self.numb_filters == 1:
                norm_adj_mat=self.__preprocess_adj_numpy(adj_mat)
            else:
                norm_adj_mat=self.__preprocess_adj_numpy_with_identity(adj_mat)
                
            obj=os.path.basename(item)
            obj=re.sub("_to_msel.*","",obj)
            class_name=obj[:obj.rfind('_')]
            class_id[idx]=self.class_mapping[class_name]

            adj_batch[idx,:,:]=norm_adj_mat
            feature_batch[idx,:,:]=feature_mat

        Y=keras.utils.to_categorical(class_id, num_classes=self.n_classes)
        return [feature_batch,adj_batch],Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            random.shuffle(self.files)

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

    def __read_shock_graph(self,sg_file):
        fid=h5py.File(sg_file,'r')

        # read in features
        feature_data=fid.get('feature')
        F_matrix=np.array(feature_data)

        # read in adj matrix
        adj_data=fid.get('adj_matrix')
        adj_matrix=np.array(adj_data)

        if self.symmetric:
            B = np.uint8(adj_matrix.transpose()) & ~np.uint8(adj_matrix)
            adj_matrix=B+adj_matrix

        # determine padding to max node size
        diff=self.numb_nodes-adj_matrix.shape[0]

        pad_adj_matrix=np.pad(adj_matrix,
                              ((0,diff),(0,diff)),
                              'constant',constant_values=(0))
        pad_F_matrix=np.pad(F_matrix,
                            ((0,diff),(0,0)),
                            'constant',constant_values=(0))
        
        return pad_adj_matrix,pad_F_matrix

    def __read_shock_graph_sparse(self,sg_file):
        fid=h5py.File(sg_file,'r')

        # read in adj matrix data

        
        sparse_adj_data=np.array(fid.get('sparse_adj_data'))
        sparse_adj_indices=np.array(fid.get('sparse_adj_indices'))
        sparse_adj_shape=np.array(fid.get('sparse_adj_shape'))

        sparse_feature_data=np.array(fid.get('sparse_feature_data'))
        sparse_feature_indices=np.array(fid.get('sparse_feature_indices'))
        sparse_feature_shape=np.array(fid.get('sparse_feature_shape'))

        adj_matrix=csr_matrix((sparse_adj_data,(sparse_adj_indices[:,0],sparse_adj_indices[:,1])),sparse_adj_shape).toarray()
        F_matrix=csr_matrix((sparse_feature_data,(sparse_feature_indices[:,0],sparse_feature_indices[:,1])),sparse_feature_shape).toarray()

        if self.symmetric:
            B = np.uint8(adj_matrix.transpose()) & ~np.uint8(adj_matrix)
            adj_matrix=B+adj_matrix

                
        return adj_matrix,F_matrix
