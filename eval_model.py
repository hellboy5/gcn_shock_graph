import dgl
import torch
import json
import sys
from tqdm import tqdm
from data.ShockGraphDataset import *
from models.gcn_sg_model import Classifier
from torch.utils.data import DataLoader
from functools import partial

def eval(model,test_dir,dataset,cache_io,symm_io,shuffle_io):

    testset=ShockGraphDataset(test_dir,dataset,cache=cache_io,symmetric=symm_io,data_augment=False)

    # Use PyTorch's DataLoader and the collate function
    # defined before.
    data_loader = DataLoader(testset, batch_size=32, shuffle=shuffle_io,
                             collate_fn=partial(collate,device_name=device))

    predicted=torch.LongTensor()
    groundtruth=torch.LongTensor()
    
    for iter, (bg, label) in enumerate(data_loader):
        output = model(bg)
        estimate = torch.max(output, 1)[1].view(-1, 1)

        predicted=torch.cat((predicted,estimate.to("cpu")),0)
        groundtruth=torch.cat((groundtruth,label.to("cpu")),0)

    groundtruth=groundtruth.view(-1)
    predicted=predicted.view(-1)

    print(groundtruth)
    print(predicted)
    print('Accuracy of argmax predictedions on the test set: {:4f}%'.format(
         (groundtruth == predicted).sum().item() / len(groundtruth) * 100))



if __name__ == "__main__":

    
    cfg=sys.argv[1]
    config_file=json.load(open(cfg))
    test_dir=config_file['test_dir']
    dataset=config_file['dataset']
    cache_io=config_file['cache_io']
    symm_io=config_file['symm_io']
    shuffle_io=config_file['shuffle_io']
    input_dim=config_file['features_dim']
    num_classes=config_file['num_classes']
    
    device_id=sys.argv[2]
    device="cuda:"+str(device_id)
    
    state_path=sys.argv[3]
    model = Classifier(input_dim, 512, num_classes,device)
    model.load_state_dict(torch.load(state_path)['model_state_dict'])
    model.to(device)
    model.eval()

    eval(model,test_dir,dataset,cache_io,symm_io,shuffle_io)
    
