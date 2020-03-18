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
from data.ShockGraphDataset_ef import *
from models.nn_sg_model import Classifier
from torch.utils.data import DataLoader
from functools import partial

def classify_data(model,data_loader,n_classes):

    predicted=np.array([],dtype=np.int32)
    groundtruth=np.array([],np.int32)
    scores=np.empty((0,n_classes))

    with torch.no_grad():
        for iter, (bg, label) in enumerate(data_loader):
            output = model(bg)
            probs_Y = torch.softmax(output, 1)
            max_scores,estimate = torch.max(probs_Y, 1)
            estimate=estimate.view(-1,1)

            predicted=np.append(predicted,estimate.to("cpu").detach().numpy())
            groundtruth=np.append(groundtruth,label.to("cpu"))
            scores=np.append(scores,probs_Y.to("cpu").detach().numpy(),axis=0)

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
    add_self_loop = config_file['self_loop']
    edge_input_dim = config_file['edge_dim']
    node_input_dim = config_file['node_dim']
    rad_scale=config_file['rad_scale']
    angle_scale=config_file['angle_scale']
    length_scale=config_file['length_scale']
    curve_scale=config_file['curve_scale']
    poly_scale=config_file['poly_scale']
    batch_io=args.batch_size
    epochs=args.epochs
    bdir=os.path.basename(train_dir)

    norm_factors={'rad_scale':rad_scale,'angle_scale':angle_scale,'length_scale':length_scale,'curve_scale':curve_scale,'poly_scale':poly_scale}
        
    prefix='m-mpnn_ni-'+str(node_input_dim)+'_nh-'+str(args.node_hidden_dim)+'_ei-'+str(edge_input_dim)+'_eh-'+str(args.edge_hidden_dim)+'_app-'+\
        str(app_io)+'_lay-'+str(args.n_layers)+'_agg-'+args.aggregate+'_res-'+str(args.residual)+'_do-'+str(args.dropout)+'_ro-'+str(args.readout)

    if args.readout == 'spp':
        extra='_ng-'+str(args.n_grid)
        prefix+=extra

    extra='-b_'+str(batch_io)
    prefix+=extra
    
    print('saving to prefix: ', prefix)
    
    # create train dataset
    testset=ShockGraphDataset(test_dir,dataset,norm_factors,app=app_io,cache=cache_io,symmetric=symm_io,data_augment=apply_da,grid=args.n_grid,self_loop=add_self_loop,dsm_norm=False)

    # Use PyTorch's DataLoader and the collate function
    # defined before.
    data_loader = DataLoader(testset, batch_size=batch_io, shuffle=shuffle_io,
                             collate_fn=partial(collate,device_name=device))

    model_files=glob.glob(prefix+'*pth')
    model_files.sort()
    
    for state_path in model_files:
        print('Using weights: ',state_path)

        model = Classifier(node_input_dim,
                           args.node_hidden_dim,
                           edge_input_dim,
                           args.edge_hidden_dim,
                           n_classes,
                           args.n_layers,
                           args.aggregate,
                           args.residual,
                           args.readout,
                           F.relu,
                           args.dropout,
                           args.n_grid,
                           device)

        model.load_state_dict(torch.load(state_path)['model_state_dict'])
        model.to(device)
        model.eval()

        groundtruth,predicted,scores=classify_data(model,data_loader,n_classes)
        confusion_matrix=np.zeros((n_classes,n_classes))
        for ind in range(0,groundtruth.shape[0]):
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
    parser = argparse.ArgumentParser(description='NN Edge Features')
    
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
    parser.add_argument("--node-hidden-dim", type=int, default=512,
                        help="number of hidden node units")
    parser.add_argument("--edge-hidden-dim", type=int, default=512,
                        help="number of hidden node units")    
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--readout", type=str, default="mean",
                        help="Readout type: mean/max/sum")
    parser.add_argument("--aggregate", type=str, default="mean",
                        help="Aggregate type: mean/max/sum")
    parser.add_argument('--residual', action="store_true",
                        help='add in residual')
    parser.add_argument("--n-grid", type=int, default=8,
                        help="number of grid cells")                                    

    args = parser.parse_args()
    print(args)

    main(args)
