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


import numpy as np
import torch
import dgl
import json
import torch.nn.functional as F
import argparse
import torch.nn as nn
from data.ShockGraphDataset_fsl import *
from models.tag_sg_sg_model import Classifier
from torch.utils.data import DataLoader
from functools import partial


def im2set(inp,layer,model,hidden_dim):

    
    embedding=torch.zeros(np.sum(inp.batch_num_nodes),hidden_dim)

    def copy_data(m,i,o):
        embedding.copy_(o.data)

    h=layer.register_forward_hook(copy_data)

    with torch.no_grad():
        h_x=model(inp)

    h.remove()
    
    return F.normalize(embedding)


def im2vec(inp,layer,model,hidden_dim):

    embedding=torch.zeros(inp.batch_size,hidden_dim)

    def copy_data(m,i,o):
        embedding.copy_(o.data)

    h=layer.register_forward_hook(copy_data)

    with torch.no_grad():
        h_x=model(inp)

    h.remove()
    
    return F.normalize(embedding)

def all_pairwise_distance(X1,X2,distance='l2'):
    
    dot_product=torch.matmul(X1,torch.t(X2))
    if distance == 'l2':
        a=torch.sum(torch.mul(X1,X1),1)
        b=torch.sum(torch.mul(X2,X2),1)
        ab=dot_product
        D=torch.sqrt(a.unsqueeze_(1)-2*ab+b.unsqueeze_(0))
        return D
    else:
        a=torch.sqrt(torch.sum(torch.mul(X1,X1),1))
        b=torch.sqrt(torch.sum(torch.mul(X2,X2),1))
        ab=dot_product
        dem=torch.ger(a,b)
        return 1.0-torch.div(ab,dem)

def predict(D,labels):
    
    indices=torch.argmin(D,dim=1)
    predicted_labels=labels[indices]
    return predicted_labels


def predict_nbnn(D,labels,nodes_per_graph,samples):

    predicted=torch.zeros(samples,dtype=torch.int32)
    classes = torch.unique_consecutive(labels)

    beg=0
    end=nodes_per_graph[0]
    for idx in range(predicted.shape[0]):
        sub_D=D[beg:end,:]
        
        class_distances=torch.zeros(len(classes))
        for jdx in range(len(classes)):
            class_exemplars=torch.where(labels==classes[jdx])
            start=class_exemplars[0][0]
            stop=class_exemplars[0][-1]+1
            min_values,indices=torch.min(sub_D[:,start:stop],dim=1)
            class_distances[jdx]=torch.sum(min_values)
            
        predicted[idx]=classes[torch.argmin(class_distances)]

        if idx < predicted.shape[0]-1:
            beg=beg+nodes_per_graph[idx]
            end=beg+nodes_per_graph[idx+1]
            
    return predicted

def main(args):

    if args.gpu<0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.gpu))

    # create dataset
    config_file=json.load(open(args.cfg))
    test_dir=config_file['test_dir']
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
    batch_io=args.batch_size
    epochs=args.epochs
    bdir=os.path.basename(train_dir)
    
    input_dim=58
    if napp:
        input_dim=input_dim+21

    if eapp:
        input_dim=input_dim+9
        
    norm_factors={'rad_scale':rad_scale,'angle_scale':angle_scale,'length_scale':length_scale,'curve_scale':curve_scale,'poly_scale':poly_scale}

    prefix='data-'+str(bdir)+':'+'mign'+'_m-tag_ni-'+str(input_dim)+'_nh-'+str(args.n_hidden)+'_lay-'+str(args.n_layers)+'_hops-'+str(args.hops)+'_napp-'+str(napp)+'_eapp-'+str(eapp)+'_do-'+str(args.dropout)+'_ro-'+str(args.readout)

    if args.readout == 'spp':
        extra='_ng-'+str(args.n_grid)
        prefix+=extra

    extra='_b-'+str(batch_io)
    prefix+=extra

    print('saving to prefix: ', prefix)
    
    # create train and test dataset
    # create train dataset
    fsl_dataset=ShockGraphDataset(test_dir,dataset,norm_factors,n_shot=args.n_shot,k_way=args.k_way,episodes=args.episodes,test_samples=args.samples,
                                  node_app=napp,edge_app=eapp,cache=True,symmetric=symm_io,data_augment=False,grid=args.n_grid)

    model_files=glob.glob(prefix+'*pth')
    model_files.sort()


    numb_train=args.n_shot*args.k_way
    
    for state_path in model_files:
        print('Using weights: ',state_path)

        model = Classifier(input_dim,
                           args.n_hidden,
                           n_classes,
                           args.n_layers,
                           args.ctype,
                           args.hops,
                           args.readout,
                           F.relu,
                           args.dropout,
                           args.n_grid,
                           args.K,
                           device)
    
        layer=nn.Module()
        for name,module in model.named_children():
            if args.nbnn:
                if name == 'layers':
                    layer=module[-1]
            else:
                if name == 'readout_fcn':
                    layer=module

        model.load_state_dict(torch.load(state_path)['model_state_dict'])
        model.to(device)
        model.eval()

        class_accuracy=np.zeros(args.episodes)
        for idx in tqdm(range(args.episodes)):

            bg,label=fsl_dataset[idx]
            
            if args.nbnn:
                embeddings=im2set(bg,layer,model,args.n_hidden)
            else:
                embeddings=im2vec(bg,layer,model,args.n_hidden)

            if args.nbnn:
                support_exemplars=np.sum(bg.batch_num_nodes[:numb_train])
            else:
                support_exemplars=numb_train
                
            train_embeddings=embeddings[:support_exemplars,:]
            if args.nbnn:
                train_labels=np.repeat(label[:numb_train],bg.batch_num_nodes[:numb_train])
            else:
                train_labels=label[:numb_train]
                
            test_embeddings=embeddings[support_exemplars:,:]
            test_labels=label[numb_train:]
            
            D=all_pairwise_distance(test_embeddings,train_embeddings,args.dist)

            if args.nbnn:
                 predicted=predict_nbnn(D,train_labels,bg.batch_num_nodes[numb_train:],test_labels.shape[0])
            else:
                predicted=predict(D,train_labels)

            groundtruth=test_labels
            
            gg=torch.sum(predicted==groundtruth)/float(len(groundtruth))
            class_accuracy[idx]=gg

        print('Class Accuracy:{:4f}%'.format(np.mean(class_accuracy)*100.0))
        del model
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphTAGConv and GraphSGConv')
    
    parser.add_argument("--cfg", type=str, default='cfg/stl10_00.json',
                        help="configuration file for the dataset")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument('--batch-size', type=int, default=64,
                        help="batch size used for training, validation and test")
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
    parser.add_argument("--ctype", type=str, default="tagconv",
                        help="Convolution type: tagconv/sgconv")                                    
    parser.add_argument("--hops", type=int, default=2,
                        help="number of hops")                                    
    parser.add_argument("--n-grid", type=int, default=8,
                        help="number of grid cells")                                    
    parser.add_argument("--flip", type=bool, default=False,
                        help="flip data")
    parser.add_argument("--K", type=float, default=100,
                        help="sort pooling keep K nodes")
    parser.add_argument("--margin", type=float, default=1.0,
                        help="margin for triplet loss")
    parser.add_argument("--neighbors", type=int, default=1,
                        help="the k in kNN")
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
    parser.add_argument("--nbnn", type=bool, default=False,
                        help="do nbnn classify")

    args = parser.parse_args()
    print(args)

    main(args)
