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
from models.gat_sg_model import Classifier
from torch.utils.data import DataLoader
from functools import partial

def classify_data(model,data_loader):


    predicted=np.array([],dtype=np.int32)
    groundtruth=np.array([],np.int32)
    scores=np.array([])

    with torch.no_grad():
        for iter, (bg, label) in enumerate(data_loader):
            output = model(bg)
            probs_Y = torch.softmax(output, 1)
            max_scores,estimate = torch.max(probs_Y, 1)
            estimate=estimate.view(-1,1)

            predicted=np.append(predicted,estimate.to("cpu").detach().numpy())
            groundtruth=np.append(groundtruth,label.to("cpu"))
            scores=np.append(scores,max_scores.to("cpu").detach().numpy())

    return groundtruth,predicted,scores

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
    app_io=config_file['app']
    symm_io=config_file['symm_io']
    shuffle_io=config_file['shuffle_io']
    n_classes=config_file['num_classes']
    apply_da=config_file['data_augment']
    num_feats = config_file['features_dim']
    batch_io=args.batch_size
    epochs=args.epochs
    bdir=os.path.basename(train_dir)
    
    prefix='gat_sg_model_'+dataset+'_'+bdir+'_'+str(args.num_layers)+'_'+str(args.num_hidden)+'_'+str(args.num_heads)+'_'+str(args.num_out_heads)

    print('saving to prefix: ', prefix)
    
    # create train dataset
    testset=ShockGraphDataset(test_dir,dataset,app=app_io,cache=True,symmetric=symm_io,
                              data_augment=False)

    # Use PyTorch's DataLoader and the collate function
    # defined before.
    data_loader = DataLoader(testset, batch_size=batch_io, shuffle=shuffle_io,
                             collate_fn=partial(collate,device_name=device))
    
    # define the model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]

    model_files=glob.glob(prefix+'*pth')
    model_files.sort()
    
    for state_path in model_files:
        print('Using weights: ',state_path)

        model = Classifier(args.num_layers,
                           num_feats,
                           args.num_hidden,
                           n_classes,
                           heads,
                           F.elu,
                           args.in_drop,
                           args.attn_drop,
                           args.alpha,
                           args.residual,
                           args.readout,
                           device)

        model.load_state_dict(torch.load(state_path)['model_state_dict'])
        model.to(device)
        model.eval()
    
        groundtruth,predicted,scores=classify_data(model,data_loader)
        confusion_matrix=np.zeros((n_classes,n_classes))
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
        print('Accuracy of argmax predictedions on the test set: {:4f}%'.format(
            (groundtruth == predicted).sum().item() / len(groundtruth) * 100))

        del model
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--cfg", type=str, default='cfg/stl10_00.json',
                        help="configuration file for the dataset")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=6,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=512,
                        help="number of hidden units")
    parser.add_argument("--residual", type=bool,default=True,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="the negative slop of leaky relu")
    parser.add_argument('--batch-size', type=int, default=64,
                        help="batch size used for training, validation and test")
    parser.add_argument("--readout", type=str, default="mean",
                        help="Readout type: mean/max/sum")   
    args = parser.parse_args()
    print(args)

    main(args)
