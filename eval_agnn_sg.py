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
    num_feats = config_file['features_dim']
    batch_io=args.batch_size
    epochs=args.epochs
    bdir=os.path.basename(train_dir)

    prefix='agnn_sg_model_'+dataset+'_'+bdir+'_'+str(args.n_layers)+'_'+str(args.n_hidden)+'_'+args.readout

    if args.readout == 'spp':
        extra='_'+str(args.n_grid)
        prefix+=extra

    # create train dataset
    testset=ShockGraphDataset(test_dir,dataset,app=app_io,cache=True,symmetric=symm_io,
                              data_augment=False,grid=args.n_grid)

    # Use PyTorch's DataLoader and the collate function
    # defined before.
    data_loader = DataLoader(testset, batch_size=batch_io, shuffle=False,
                             collate_fn=partial(collate,device_name=device))

    if args.flip:
        testset_flip=ShockGraphDataset(test_dir,dataset,app=app_io,cache=True,
                                       symmetric=symm_io,data_augment=False,grid=args.n_grid,flip_pp=True)
        
        # Use PyTorch's DataLoader and the collate function
        # defined before.
        data_loader_flip = DataLoader(testset_flip, batch_size=batch_io, shuffle=False,
                                      collate_fn=partial(collate,device_name=device))


    print('A AGNNConv Graph Classifier is being trained')    

    model_files=glob.glob(prefix+'*pth')
    model_files.sort()
    
    for state_path in model_files:
        print('Using weights: ',state_path)

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


        model.load_state_dict(torch.load(state_path)['model_state_dict'])
        model.to(device)
        model.eval()

        groundtruth,predicted,scores=classify_data(model,data_loader,n_classes)

        if args.flip:
            _,predicted_flip,scores_flip=classify_data(model,data_loader_flip,n_classes)

        combined_predicted=np.copy(predicted)
                
        if args.flip:
            for i in range(groundtruth.shape[0]):
                if np.max(scores_flip[i,:]) > np.max(scores[i,:]):
                    print("flipping: ",np.max(scores_flip[i,:]),np.max(scores[i,:]),predicted[i],predicted_flip[i])
                    combined_predicted[i]=predicted_flip[i]

        confusion_matrix=np.zeros((n_classes,n_classes))
        for ind in range(0,groundtruth.shape[0]):
            if groundtruth[ind]==combined_predicted[ind]:
                confusion_matrix[groundtruth[ind],groundtruth[ind]]+=1
            else:
                confusion_matrix[groundtruth[ind],combined_predicted[ind]]+=1

        confusion_matrix=(confusion_matrix/np.sum(confusion_matrix,1)[:,None])*100

        #print(confusion_matrix)

        mAP=np.diagonal(confusion_matrix)

        print(mAP)
        print("mAP: ",np.mean(mAP))
        print('Accuracy of argmax combined_predictedions on the test set: {:4f}%'.format(
            (groundtruth == combined_predicted).sum().item() / len(groundtruth) * 100))

        testfiles=testset.files
        fid=open('output.txt','w')
        for ind in range(0,len(testfiles)):
            line=[testfiles[ind]+' ',str(groundtruth[ind])+' ',str(combined_predicted[ind])+' ']
            for idx in range(scores.shape[1]):
                val=scores[ind,idx]
                if idx==scores.shape[1]-1:
                    line.append(str(val)+'\n')
                else:
                    line.append(str(val)+' ')
            
            fid.writelines(line)

        fid.close()

        del model
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphAGNNConv')
    
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
    parser.add_argument("--flip", type=bool, default=False,
                        help="flip data")                                    

    args = parser.parse_args()
    print(args)

    main(args)
