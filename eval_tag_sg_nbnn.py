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
from data.ShockGraphDataset import *
from models.tag_sg_sg_model import Classifier
from torch.utils.data import DataLoader
from functools import partial

def im2vec(inp,layer,model,hidden_dim):

    embedding=torch.zeros(16,hidden_dim)

    def copy_data(m,i,o):
        o=torch.squeeze(o)
        numb_elements=o.shape[1]*o.shape[2]
        hg=torch.flatten(o,1)
        vecs=torch.reshape(torch.t(hg),(numb_elements,o.shape[0]))
        embedding.copy_(vecs)

    h=layer.register_forward_hook(copy_data)

    with torch.no_grad():
        h_x=model(inp)

    h.remove()

    return F.normalize(embedding)

def cosine_distance(X1,X2):
    return 1.0-torch.matmul(X1,torch.t(X2))

def predict(D,labels,n_classes,spp):

    values=D.shape[0]/spp
    predicted=torch.zeros(int(D.shape[0]/spp),dtype=torch.int32)
    
    for idx in range(predicted.shape[0]):
        beg=idx*spp
        end=beg+spp
        sub_D=D[beg:end,:]

        class_distances=torch.zeros(n_classes)
        for jdx in range(n_classes):
            class_exemplars=torch.where(labels==jdx)
            start=class_exemplars[0][0]
            stop=class_exemplars[0][-1]+1
            min_values,indices=torch.min(sub_D[:,start:stop],dim=1)
            class_distances[jdx]=torch.sum(min_values)
            
        predicted[idx]=torch.argmin(class_distances)
        
    return predicted

def main(args):

    if args.gpu<0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.gpu))

    # create dataset
    config_file=json.load(open(args.cfg))
    train_dir=config_file['train_dir']
    test_dir=config_file['test_dir']
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
    bdir='se_tcg_train_fc'
    
    input_dim=58
    if napp:
        input_dim=input_dim+21

    if eapp:
        input_dim=input_dim+9
        
    norm_factors={'rad_scale':rad_scale,'angle_scale':angle_scale,'length_scale':length_scale,'curve_scale':curve_scale,'poly_scale':poly_scale}

    prefix='data-'+str(bdir)+':'+str(dataset)+'_m-tag_ni-'+str(input_dim)+'_nh-'+str(args.n_hidden)+'_lay-'+str(args.n_layers)+'_hops-'+str(args.hops)+'_napp-'+str(napp)+'_eapp-'+str(eapp)+'_do-'+str(args.dropout)+'_ro-'+str(args.readout)

    if args.readout == 'spp':
        extra='_ng-'+str(args.n_grid)
        prefix+=extra

    extra='_b-'+str(batch_io)
    prefix+=extra

    print('saving to prefix: ', prefix)
    
    # create train and test dataset
    trainset=ShockGraphDataset(train_dir,dataset,norm_factors,node_app=napp,edge_app=eapp,cache=True,symmetric=symm_io,
                               data_augment=False,grid=args.n_grid)
    
    testset=ShockGraphDataset(test_dir,dataset,norm_factors,node_app=napp,edge_app=eapp,cache=True,symmetric=symm_io,
                              data_augment=False,grid=args.n_grid)

    # Use PyTorch's DataLoader and the collate function
    # defined before.
    data_loader_train = DataLoader(trainset, batch_size=1, shuffle=False,
                                   collate_fn=partial(collate,device_name=device))
    data_loader_test  = DataLoader(testset, batch_size=1, shuffle=False,
                                   collate_fn=partial(collate,device_name=device))    

    model_files=glob.glob(prefix+'*pth')
    model_files.sort()

    spp=args.n_grid*args.n_grid
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
            if name == 'readout_fcn':
                layer=module

        model.load_state_dict(torch.load(state_path)['model_state_dict'])
        model.to(device)
        model.eval()

        # get train embeddings and labels
        train_embeddings=torch.zeros((len(data_loader_train)*spp,args.n_hidden))
        train_labels=torch.zeros(len(data_loader_train)*spp,dtype=torch.int32)
        start=0
        stop=spp
        for iter, (bg, label) in enumerate(data_loader_train):
            train_embeddings[start:stop,:] = im2vec(bg,layer,model,args.n_hidden)
            train_labels[start:stop]       = label
            start=stop
            stop=start+spp
            
        # get test embeddings and labels
        test_embeddings=torch.zeros((len(data_loader_test)*spp,args.n_hidden))
        test_labels=torch.zeros(len(data_loader_test),dtype=torch.int32)
        start=0
        stop=spp
        for iter, (bg, label) in enumerate(data_loader_test):
            test_embeddings[start:stop,:] = im2vec(bg,layer,model,args.n_hidden)
            test_labels[iter]             = label
            start=stop
            stop=start+spp

        D=cosine_distance(test_embeddings,train_embeddings)
                
        predicted=predict(D,train_labels,n_classes,spp)
        groundtruth=test_labels
        
        confusion_matrix=np.zeros((n_classes,n_classes))
        for ind in range(groundtruth.shape[0]):
            if groundtruth[ind]==predicted[ind]:
                confusion_matrix[groundtruth[ind],groundtruth[ind]]+=1
            else:
                confusion_matrix[groundtruth[ind],predicted[ind]]+=1

        confusion_matrix=(confusion_matrix/np.sum(confusion_matrix,1)[:,None])*100

        #print(confusion_matrix)

        mAP=np.diagonal(confusion_matrix)

        print(mAP)
        print("mAP: ",np.mean(mAP))
        print('Accuracy of argmax predictedions on the test set: {:4f}%'.format(
            (groundtruth == predicted).sum().item() / len(groundtruth) * 100))

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

    args = parser.parse_args()
    print(args)

    main(args)
