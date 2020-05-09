"""
Graph Attention Networks (PPI Dataset) in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
Compared with the original paper, this code implements
early stopping.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import os
import numpy as np
import torch
import dgl
import json
import torch.nn.functional as F
import argparse
import torch.nn as nn
from data.ShockGraphDataset_fsl import *
from models.embedding_network import Classifier
from torch.utils.data import DataLoader
from functools import partial


def cosine_similarity(X1,X2):
    
    dot_product=torch.matmul(X1,torch.t(X2))
    a=torch.sqrt(torch.sum(torch.mul(X1,X1),1))
    b=torch.sqrt(torch.sum(torch.mul(X2,X2),1))
    ab=dot_product
    dem=torch.ger(a,b)
    return torch.div(ab,dem)

def predict(D,labels):

    classes = torch.unique_consecutive(labels)
    d2c=torch.zeros(D.shape[0],len(classes),dtype=torch.float32)
    
    for idx in range(d2c.shape[0]):
        sub_D=D[idx,:]
        class_distances=torch.zeros(len(classes))
        for jdx in range(len(classes)):
            class_exemplars=torch.where(labels==classes[jdx])
            start=class_exemplars[0][0]
            stop=class_exemplars[0][-1]+1
            max_value=torch.max(sub_D[start:stop])
            class_distances[jdx]=max_value
            
        d2c[idx,:]=class_distances
            
    return d2c


def predict_nbnn(D,labels,nodes_per_graph,samples):

    classes = torch.unique_consecutive(labels)
    d2c=torch.zeros(samples,len(classes),dtype=torch.float32)
    
    beg=0
    end=nodes_per_graph[0]
    for idx in range(d2c.shape[0]):
        sub_D=D[beg:end,:]
        
        class_distances=torch.zeros(len(classes))
        for jdx in range(len(classes)):
            class_exemplars=torch.where(labels==classes[jdx])
            start=class_exemplars[0][0]
            stop=class_exemplars[0][-1]+1
            max_values,indices=torch.max(sub_D[:,start:stop],dim=1)
            class_distances[jdx]=torch.sum(max_values)
            
        d2c[idx,:]=class_distances

        if idx < d2c.shape[0]-1:
            beg=beg+nodes_per_graph[idx]
            end=beg+nodes_per_graph[idx+1]
            
    return d2c



def main(args):

    if args.gpu<0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.gpu))

    # create dataset
    config_file=json.load(open(args.cfg))
    train_dir=config_file['train_dir']
    dataset=config_file['dataset']
    cache_io=config_file['cache_io']
    napp=config_file['node_app']
    eapp=config_file['edge_app']
    symm_io=config_file['symm_io']
    shuffle_io=config_file['shuffle_io']
    n_classes=config_file['num_classes']
    apply_da=config_file['data_augment']
    rad_scale=config_file['rad_scale']
    angle_scale=config_file['angle_scale']
    length_scale=config_file['length_scale']
    curve_scale=config_file['curve_scale']
    poly_scale=config_file['poly_scale']
    bdir=os.path.basename(train_dir)
    
    input_dim=58
    if napp:
        input_dim=input_dim+21

    if eapp:
        input_dim=input_dim+9
        
    norm_factors={'rad_scale':rad_scale,'angle_scale':angle_scale,'length_scale':length_scale,'curve_scale':curve_scale,'poly_scale':poly_scale}

    prefix='data-'+str(bdir)+':'+str(dataset)+'_m-tag_ni-'+str(input_dim)+'_nh-'+str(args.n_hidden)+'_lay-'+str(args.n_layers)+'_napp-'+str(napp)+'_eapp-'+str(eapp)+'_ro-'+str(args.readout)
    prefix+='_loc-'+str(args.local)+'_nshot-'+str(args.n_shot)+'_kway-'+str(args.k_way)+'_samp-'+str(args.samples)+'_epi-'+str(args.episodes)+'_norm-'+str(args.norm)

    if args.local == False:
        prefix += '_emb-'+str(args.embed_dim)
        
    if args.readout == 'spp':
        extra='_ng-'+str(args.n_grid)
        prefix+=extra

    print('saving to prefix: ', prefix)

    # create train dataset
    trainset=ShockGraphDataset(train_dir,dataset,norm_factors,n_shot=args.n_shot,k_way=args.k_way,episodes=args.episodes,test_samples=args.samples,
                               node_app=napp,edge_app=eapp,cache=True,symmetric=symm_io,data_augment=False,grid=args.n_grid)

    print('A TAGConv Graph Classifier is being trained')

    model = Classifier(input_dim,
                       args.n_hidden,
                       args.embed_dim,
                       args.n_layers,
                       args.hops,
                       args.readout,
                       F.relu,
                       args.dropout,
                       args.local,
                       args.norm,
                       args.n_grid,
                       args.K,
                       device)

    loss_func = nn.CrossEntropyLoss()
     
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model = model.to(device)
    model.train()
    
    print(model)

    numb_train=args.n_shot*args.k_way
        
    avg_loss = 0
    for idx in range(args.episodes):
        bg,label=trainset[idx]
        
        embeddings = model(bg)

        if args.local:
            support_exemplars=np.sum(bg.batch_num_nodes[:numb_train])
        else:
            support_exemplars=numb_train

        train_embeddings=embeddings[:support_exemplars,:]
        if args.local:
            train_labels=np.repeat(label[:numb_train],bg.batch_num_nodes[:numb_train])
        else:
            train_labels=label[:numb_train]
                
        test_embeddings=embeddings[support_exemplars:,:]
        _,test_labels=torch.unique_consecutive(label[numb_train:],return_inverse=True)
            
        D=cosine_similarity(test_embeddings,train_embeddings)

        if args.local:
            prediction=predict_nbnn(D,train_labels,bg.batch_num_nodes[numb_train:],test_labels.shape[0])
        else:
            prediction=predict(D,train_labels)

        loss = loss_func(prediction,test_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        episode_loss = loss.detach().item()
        avg_loss = (episode_loss+avg_loss)

        print('Episode {}, Loss {:.6f}, Loss avg {:.6f}'.format(idx, episode_loss,avg_loss/(idx+1)))

        if (idx+1) % 25 == 0:
            path=prefix+'_episode_'+str(idx+1).zfill(4)+'.pth'
            torch.save({
                'episode': idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': episode_loss}, path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphTAGConv and GraphSGConv')
    
    parser.add_argument("--cfg", type=str, default='cfg/stl10_00.json',
                        help="configuration file for the dataset")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-hidden", type=int, default=512,
                        help="number of hidden gsage units")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--readout", type=str, default="mean",
                        help="Readout type: mean/max/sum")
    parser.add_argument("--hops", type=int, default=2,
                        help="number of hops")                                    
    parser.add_argument("--n-grid", type=int, default=8,
                        help="number of grid cells")                                    
    parser.add_argument("--K", type=float, default=100,
                        help="sort pooling keep K nodes")
    parser.add_argument("--embed_dim", type=int, default="192",
                        help="dim to embed")
    parser.add_argument("--n_shot", type=int, default=5,
                        help="N shot per class")
    parser.add_argument("--k_way", type=int, default=5,
                        help="K way classes")
    parser.add_argument("--episodes", type=int, default=600,
                        help="number of episodes")
    parser.add_argument("--samples", type=int, default=15,
                        help="samples")
    parser.add_argument("--dist", type=str,default='cos',
                        help="should i use l2/coss for distance? ")
    parser.add_argument("--norm", type=bool, default=False,
                        help="normalize or not")
    parser.add_argument("--local", type=bool, default=False,
                        help="do nbnn classify")

    args = parser.parse_args()
    print(args)

    main(args)
