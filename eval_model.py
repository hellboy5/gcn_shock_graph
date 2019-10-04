import dgl
import torch
import torch.nn as nn
import json
import sys
from tqdm import tqdm
from data.ShockGraphDataset import *
from models.gcn_sg_model import Classifier
from torch.utils.data import DataLoader
from functools import partial

def classify_data(model,testset,batch_io,device):

    # Use PyTorch's DataLoader and the collate function
    # defined before.
    data_loader = DataLoader(testset, batch_size=batch_io, shuffle=False,
                             collate_fn=partial(collate,device_name=device))

    testfiles=testset.files
    predicted=np.array([],dtype=np.int32)
    groundtruth=np.array([],np.int32)
    scores=np.array([])
    
    for iter, (bg, label) in enumerate(data_loader):
        output = model(bg)
        probs_Y = torch.softmax(output, 1)
        max_scores,estimate = torch.max(probs_Y, 1)
        estimate=estimate.view(-1,1)

        predicted=np.append(predicted,estimate.to("cpu").detach().numpy())
        groundtruth=np.append(groundtruth,label.to("cpu"))
        scores=np.append(scores,max_scores.to("cpu").detach().numpy())

    return groundtruth,predicted,scores,testfiles

def eval(config_file,state_path,device,flip):

    
    test_dir=config_file['test_dir']
    dataset=config_file['dataset']
    cache_io=True
    symm_io=config_file['symm_io']
    shuffle_io=False
    num_classes=config_file['num_classes']
    input_dim=config_file['features_dim']
    hidden_dim=config_file['hidden_dim']
    hidden_layers=config_file['hidden_layers']
    aggregate=config_file['aggregate']
    combine=config_file['combine']
    batch_io=config_file['batch']
    dropout=config_file['dropout']

    model = Classifier(input_dim, hidden_dim, num_classes,hidden_layers,combine,nn.functional.relu,
                       dropout,device)
    model.load_state_dict(torch.load(state_path)['model_state_dict'])
    model.to(device)
    model.eval()

    testset=ShockGraphDataset(test_dir,dataset,cache=cache_io,symmetric=symm_io,data_augment=False)
    if flip:
        testset_flip=ShockGraphDataset(test_dir,dataset,cache=cache_io,symmetric=symm_io,data_augment=False,flip_pp=True)


    groundtruth,predicted,scores,testfiles=classify_data(model,testset,batch_io,device)

    if flip:
        _,predicted_flip,scores_flip,_=classify_data(model,testset_flip,batch_io,device)

    confusion_matrix=np.zeros(( num_classes, num_classes))

    combined_predicted=np.copy(predicted)

    if flip:
        for i in range(groundtruth.shape[0]):
            if scores_flip[i] > scores[i]:
                print("flipping: ",scores_flip[i],scores[i],combined_predicted[i],predicted_flip[i])
                combined_predicted[i]=predicted_flip[i]
                                        
    for ind in range(0,groundtruth.shape[0]):

        if groundtruth[ind]==combined_predicted[ind]:
            confusion_matrix[groundtruth[ind],groundtruth[ind]]+=1
        else:
            confusion_matrix[groundtruth[ind],combined_predicted[ind]]+=1

    confusion_matrix=(confusion_matrix/np.sum(confusion_matrix,1)[:,None])*100

    print(confusion_matrix)

    mAP=np.diagonal(confusion_matrix)

    print(mAP)
    print("mAP: ",np.mean(mAP))
    print(groundtruth)
    print(predicted)
    print(combined_predicted)
    print('Accuracy of argmax combined_predictedions on the test set: {:4f}%'.format(
         (groundtruth == combined_predicted).sum().item() / len(groundtruth) * 100))

    fid=open('output.txt','w')
    for ind in range(0,len(testfiles)):
        line=[testfiles[ind]+' ',str(groundtruth[ind])+' ',str(predicted[ind])+' ',str(scores[ind])+'\n']
        fid.writelines(line)

    fid.close()

if __name__ == "__main__":

    
    cfg=sys.argv[1]
    config_file=json.load(open(cfg))

    device_id=sys.argv[2]
    device="cuda:"+str(device_id)
    
    state_path=sys.argv[3]

    flip=sys.argv[4]
    eval(config_file,state_path,device,int(flip))
    
