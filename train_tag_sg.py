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
from data.ShockGraphDataset import *
from models.tag_sg_sg_model import Classifier
from torch.utils.data import DataLoader
from functools import partial

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
    domains=config_file['domains']
    batch_io=args.batch_size
    epochs=args.epochs
    bdir=os.path.basename(train_dir)
    print(domains)
    
    input_dim=58
    if napp:
        input_dim=input_dim+21

    if eapp:
        input_dim=input_dim+9
        
    norm_factors={'rad_scale':rad_scale,'angle_scale':angle_scale,'length_scale':length_scale,'curve_scale':curve_scale,'poly_scale':poly_scale}

    prefix='data-'+str(bdir)+':'+str(dataset)+'_m-tag_ni-'+str(input_dim)+'_nh-'+str(args.n_hidden)+'_lay-'+str(args.n_layers)+'_hops-'+str(args.hops)+'_napp-'+str(napp)+'_eapp-'+str(eapp)+'_do-'+str(args.dropout)+'_ro-'+str(args.readout)+'_norm-'+str(args.norm)

    if args.readout == 'spp':
        extra='_ng-'+str(args.n_grid)
        prefix+=extra

    extra='_b-'+str(batch_io)
    prefix+=extra

    print('saving to prefix: ', prefix)
    
    # create train dataset
    trainset=ShockGraphDataset(train_dir,dataset,norm_factors,node_app=napp,edge_app=eapp,cache=cache_io,symmetric=symm_io,data_augment=apply_da,grid=args.n_grid,
                               domains=domains)

    # Use PyTorch's DataLoader and the collate function
    # defined before.
    data_loader = DataLoader(trainset, batch_size=batch_io, shuffle=shuffle_io,
                             collate_fn=partial(collate,device_name=device))

    if args.ctype == 'tagconv':
        print('A TAGConv Graph Classifier is being trained')
    else:
        print('A SGConv Graph Classifier is being trained')

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
                       args.norm,
                       device)
        
    loss_func = nn.CrossEntropyLoss()

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model = model.to(device)
    model.train()
    
    print(model)
    
    epoch_losses = []
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for iter, (bg, label) in enumerate(data_loader):            
            prediction = model(bg)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        epoch_losses.append(epoch_loss)
        print('Epoch {}, loss {:.6f}'.format(epoch, epoch_loss))

        if (epoch+1) % 25 == 0:
            path=prefix+'_epoch_'+str(epoch+1).zfill(3)+'.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss}, path)




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
    parser.add_argument("--K", type=float, default=100,
                        help="sort pooling keep K nodes")                                    
    parser.add_argument("--norm", type=str, default='batch',
                        help="type of normalization (batch/layer/graph/instance/none)")                                    

    args = parser.parse_args()
    print(args)

    main(args)
