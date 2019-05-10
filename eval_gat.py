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
from models.gat_sg_model import GAT
from torch.utils.data import DataLoader
from functools import partial

def main(args):
    if args.gpu<0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.gpu))



    # create dataset
    config_file=json.load(open(args.cfg))
    test_dir=config_file['test_dir']
    dataset=config_file['dataset']
    cache_io=config_file['cache_io']
    symm_io=config_file['symm_io']
    app_io=config_file['app']
    shuffle_io=False
    n_classes=config_file['num_classes']
    apply_da=False
    num_feats = config_file['features_dim']
    batch_io=args.batch_size
    state_path=args.model
    
    # create train dataset
    testset=ShockGraphDataset(test_dir,dataset,app=app_io,cache=cache_io,symmetric=symm_io,data_augment=apply_da)

    # Use PyTorch's DataLoader and the collate function
    # defined before.
    data_loader = DataLoader(testset, batch_size=batch_io, shuffle=shuffle_io,
                             collate_fn=partial(collate,device_name=device))
    
    # define the model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]

    model = GAT(args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.alpha,
                args.residual,
                device)


     
    # load pre trained weights
    model.load_state_dict(torch.load(state_path)['model_state_dict'])
    model = model.to(device)
    model.train()
    
    print(model)

    predicted=torch.LongTensor()
    groundtruth=torch.LongTensor()
    confusion_matrix=np.zeros((n_classes,n_classes))
    for iter, (bg, label) in enumerate(data_loader):
        output = model(bg)
        estimate = torch.max(output, 1)[1].view(-1, 1)

        predicted=torch.cat((predicted,estimate.to("cpu")),0)
        groundtruth=torch.cat((groundtruth,label.to("cpu")),0)
    
    groundtruth=groundtruth.view(-1)
    predicted=predicted.view(-1)

    for ind in range(0,groundtruth.shape[0]):
        if groundtruth[ind]==predicted[ind]:
            confusion_matrix[groundtruth[ind],groundtruth[ind]]+=1
        else:
            confusion_matrix[groundtruth[ind],predicted[ind]]+=1

    confusion_matrix=(confusion_matrix/np.sum(confusion_matrix,1)[:,None])*100

    print(confusion_matrix)

    mAP=np.diagonal(confusion_matrix)

    print(mAP)
    print("mAP: ",np.mean(mAP))
    print(groundtruth)
    print(predicted)
    print('Accuracy of argmax predictedions on the test set: {:4f}%'.format(
         (groundtruth == predicted).sum().item() / len(groundtruth) * 100))





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--cfg", type=str, default='cfg/stl10_00.json',
                        help="configuration file for the dataset")
    parser.add_argument("--model", type=str,
                        help="load pretrained model/weights"), 
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=6,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=256,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0,
                        help="attention dropout")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="the negative slop of leaky relu")
    parser.add_argument('--batch-size', type=int, default=64,
                        help="batch size used for training, validation and test")
    args = parser.parse_args()
    print(args)

    main(args)
