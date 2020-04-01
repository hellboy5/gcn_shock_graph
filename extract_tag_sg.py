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

    embedding=torch.zeros(hidden_dim)

    def copy_data(m,i,o):
        embedding.copy_(o.data[0,:])

    h=layer.register_forward_hook(copy_data)

    with torch.no_grad():
        h_x=model(inp)

    h.remove()

    return embedding

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
    rad_scale=config_file['rad_scale']
    angle_scale=config_file['angle_scale']
    length_scale=config_file['length_scale']
    curve_scale=config_file['curve_scale']
    poly_scale=config_file['poly_scale']
    batch_io=1
    epochs=args.epochs
    bdir=os.path.basename(train_dir)

    norm_factors={'rad_scale':rad_scale,'angle_scale':angle_scale,'length_scale':length_scale,'curve_scale':curve_scale,'poly_scale':poly_scale}

    prefix=args.ctype+'_sg_model_'+dataset+'_'+bdir+'_'+str(args.n_layers)+'_'+str(args.n_hidden)+'_'+str(args.hops)+'_'+args.readout+'_'+str(args.dropout)+'_'+str(app_io)

    if args.readout == 'spp':
        extra='_'+str(args.n_grid)
        prefix+=extra

    # create train dataset
    testset=ShockGraphDataset(train_dir,dataset,norm_factors,app=app_io,cache=True,symmetric=symm_io,data_augment=False,grid=args.n_grid)

    # Use PyTorch's DataLoader and the collate function
    # defined before.
    data_loader = DataLoader(testset, batch_size=batch_io, shuffle=False,
                             collate_fn=partial(collate,device_name=device))

    if args.ctype == 'tagconv':
        print('A TAGConv Graph Classifier is being trained')
    else:
        print('A SGConv Graph Classifier is being trained')


    model = Classifier(num_feats,
                       args.n_hidden,
                       n_classes,
                       args.n_layers,
                       args.ctype,
                       args.hops,
                       args.readout,
                       F.relu,
                       args.dropout,
                       args.n_grid,
                       device)
    

    model.load_state_dict(torch.load(args.model)['model_state_dict'])


    layer=nn.Module()
    for name,module in model.named_children():
        if name == 'readout_fcn':
            layer=module

    model.to(device)
    model.eval()

    embeddings=np.zeros((len(data_loader),args.n_hidden))
    for iter, (bg, label) in enumerate(data_loader):
        embeddings[iter,:] = im2vec(bg,layer,model,args.n_hidden)
        

    np.savetxt('test_embeddings.txt',embeddings,delimiter=' ')


    fid=open('testset.txt','w')
    for i in testset.files:
        fid.writelines(i+'\n')
    fid.close()
    
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
                        help="number of grid cells"),
    parser.add_argument("--model", type=str,
                        help="load pretrained model/weights")

    args = parser.parse_args()
    print(args)

    main(args)
