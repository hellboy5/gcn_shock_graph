import numpy as np
import keras 
import glob
import random
import h5py
import os
import re

class ShockGraphDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,directory, labels, numb_nodes,numb_attrs,batch_size=32,
                 n_classes=10, shuffle=True):
        'Initialization'
        
        self.directory = directory
        self.labels = labels
        self.numb_nodes = numb_nodes
        self.numb_attrs = numb_attrs
        self.batch_size=batch_size
        self.n_classes = n_classes
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
        adj_batch=np.zeros((self.batch_size,self.numb_nodes,self.numb_nodes))
        feature_batch=np.zeros((self.batch_size,self.numb_nodes,self.numb_attrs))

        for idx in range(0,len(batch_indices)):
            item=self.files[batch_indices[idx]]
            adj_mat,feature_mat=self.__read_shock_graph(item)

            norm_adj_mat=self.__preprocess_adj_numpy(adj_mat)
            
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

        # determine padding to max node size
        diff=self.numb_nodes-adj_matrix.shape[0]

        pad_adj_matrix=np.pad(adj_matrix,
                              ((0,diff),(0,diff)),
                              'constant',constant_values=(0))
        pad_F_matrix=np.pad(F_matrix,
                            ((0,diff),(0,0)),
                            'constant',constant_values=(0))
        
        return pad_adj_matrix,pad_F_matrix
