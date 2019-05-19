import dgl
import torch
import json
import sys
import torch.nn as nn
from tqdm import tqdm
from data.ShockGraphDataset import *
from models.gcn_sg_model import Classifier
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
import os

def train(config_file,device):

    train_dir=config_file['train_dir']
    dataset=config_file['dataset']
    cache_io=config_file['cache_io']
    symm_io=config_file['symm_io']
    shuffle_io=config_file['shuffle_io']
    num_classes=config_file['num_classes']
    input_dim=config_file['features_dim']
    hidden_dim=config_file['hidden_dim']
    hidden_layers=config_file['hidden_layers']
    aggregate=config_file['aggregate']
    combine=config_file['combine']
    epochs=config_file['epochs']
    batch_io=config_file['batch']
    apply_da=config_file['data_augment']
    bdir=os.path.basename(train_dir)
    
    prefix='gcn_sg_model_app_'+dataset+'_'+bdir+'_'+aggregate+'_'+combine+'_'+str(hidden_dim)+'_'+str(hidden_layers)
    
    print("Training with batch size of: ",batch_io," over ",epochs," Epochs with da: ",apply_da)
    print("Writing out to : ",prefix)
    
    trainset=ShockGraphDataset(train_dir,dataset,cache=cache_io,symmetric=symm_io,data_augment=apply_da)


    
    # Use PyTorch's DataLoader and the collate function
    # defined before.
    data_loader = DataLoader(trainset, batch_size=batch_io, shuffle=shuffle_io,
                             collate_fn=partial(collate,device_name=device))
    
    # Create model
    model = Classifier(input_dim, hidden_dim, num_classes,hidden_layers,aggregate,combine,device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)#,weight_decay=0.00005/2.0)
    model.to(device)
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

        if (epoch+1) % 50 == 0:
            path=prefix+'_epoch_'+str(epoch+1).zfill(3)+'.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss}, path)

        





if __name__ == "__main__":


    
    cfg=sys.argv[1]
    config_file=json.load(open(cfg))

    device_id=sys.argv[2]
    device="cuda:"+str(device_id)
    train(config_file,device)
    
