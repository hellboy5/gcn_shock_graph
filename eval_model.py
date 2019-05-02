import dgl
import torch
import json
import sys
from tqdm import tqdm
from data.ShockGraphDataset import *
from models.gcn_sg_model import Classifier
from torch.utils.data import DataLoader
from functools import partial

def eval(config_file,state_path,device):

    
    test_dir=config_file['test_dir']
    dataset=config_file['dataset']
    cache_io=config_file['cache_io']
    symm_io=config_file['symm_io']
    shuffle_io=False
    num_classes=config_file['num_classes']
    input_dim=config_file['features_dim']
    hidden_dim=config_file['hidden_dim']
    hidden_layers=config_file['hidden_layers']
    aggregate=config_file['aggregate']
    combine=config_file['combine']
    batch_io=config_file['batch']

    model = Classifier(input_dim, hidden_dim, num_classes,hidden_layers,aggregate,combine,device)
    model.load_state_dict(torch.load(state_path)['model_state_dict'])
    model.to(device)
    model.eval()

    testset=ShockGraphDataset(test_dir,dataset,cache=cache_io,symmetric=symm_io,data_augment=False)

    # Use PyTorch's DataLoader and the collate function
    # defined before.
    data_loader = DataLoader(testset, batch_size=batch_io, shuffle=False,
                             collate_fn=partial(collate,device_name=device))

    predicted=torch.LongTensor()
    groundtruth=torch.LongTensor()
    confusion_matrix=np.zeros((num_classes,num_classes))
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



if __name__ == "__main__":

    
    cfg=sys.argv[1]
    config_file=json.load(open(cfg))

    device_id=sys.argv[2]
    device="cuda:"+str(device_id)
    
    state_path=sys.argv[3]

    eval(config_file,state_path,device)
    
