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
from models.agnn_sg_model import Classifier
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
    app_io=config_file['app']
    symm_io=config_file['symm_io']
    shuffle_io=config_file['shuffle_io']
    n_classes=config_file['num_classes']
    apply_da=config_file['data_augment']
    num_feats = config_file['features_dim']
    batch_io=args.batch_size
    epochs=args.epochs
    bdir=os.path.basename(train_dir)
    
    prefix='agnn_sg_model_'+dataset+'_'+bdir+'_'+str(args.n_layers)+'_'+str(args.n_hidden)+'_'+args.readout

    if args.readout == 'spp':
        extra='_'+str(args.n_grid)
        prefix+=extra

    print('saving to prefix: ', prefix)
    
    # create train dataset
    trainset=ShockGraphDataset(train_dir,dataset,app=app_io,cache=cache_io,symmetric=symm_io,data_augment=apply_da,grid=args.n_grid)

    # Use PyTorch's DataLoader and the collate function
    # defined before.
    data_loader = DataLoader(trainset, batch_size=batch_io, shuffle=shuffle_io,
                             collate_fn=partial(collate,device_name=device))

    print('A AGNN Graph Classifier is being trained')
    
    model = Classifier(num_feats,
                       args.n_hidden,
                       n_classes,
                       args.n_layers,
                       args.init_beta,
                       args.learn_beta,
                       args.readout,
                       F.relu,
                       args.dropout,
                       args.n_grid,
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
    parser.add_argument("--init-beta", type=float, default=1.0,
                        help="Init beta to start learning")
    parser.add_argument('--learn-beta', action="store_true",
                        help='learn the beta weighting')
    parser.add_argument("--n-grid", type=int, default=8,
                        help="number of grid cells")                                    

    args = parser.parse_args()
    print(args)

    main(args)
