import os
import sys
import numpy as np
import math
import torch

from scipy.spatial.distance import cdist


def pairwise_distance(X1,X2,distance='euclid'):

    D=all_pairwise_distance(X1,X2,distance)
    return torch.diag(D)
    
def all_pairwise_distance(X1,X2,distance='euclid'):

    dot_product=torch.matmul(X1,torch.t(X2))
    if distance == 'euclid':
        a=torch.sum(torch.mul(X1,X1),1)
        b=torch.sum(torch.mul(X2,X2),1)
        ab=dot_product
        D=torch.sqrt(a.unsqueeze_(1)-2*ab+b.unsqueeze_(0))
        return D
    else:
        a=torch.sqrt(torch.sum(torch.mul(X1,X1),1))
        b=torch.sqrt(torch.sum(torch.mul(X2,X2),1))
        ab=dot_product
        dem=torch.ger(a,b)
        return 1.0-torch.div(ab,dem)
    

test=torch.randn(100,192)
train=torch.randn(150,192)

D_l2_pred1=all_pairwise_distance(test,train)
D_l2_pred2=all_pairwise_distance(train,test)

D_l2_gt1=cdist(test,train)
D_l2_gt2=cdist(train,test)


print(torch.max(torch.abs(D_l2_pred1-D_l2_gt1)))
print(torch.max(torch.abs(D_l2_pred2-D_l2_gt2)))


D_cos_pred1=all_pairwise_distance(test,train,'cosine')
D_cos_pred2=all_pairwise_distance(train,test,'cosine')

D_cos_gt1=cdist(test,train,'cosine')
D_cos_gt2=cdist(train,test,'cosine')

print(D_l2_pred1.shape)
print(D_l2_pred2.shape)
print(D_cos_pred1.shape)
print(D_cos_pred2.shape)

print(D_l2_pred1[1,0:10])
print(D_l2_gt1[1,0:10])

print(torch.max(torch.abs(D_cos_pred1-D_cos_gt1)))
print(torch.max(torch.abs(D_cos_pred2-D_cos_gt2)))

print('doing pairwise')

test=torch.randn(75,75)
train=torch.randn(75,75)

D_l2_pred1=pairwise_distance(test,train)
D_l2_pred2=pairwise_distance(train,test)

D_l2_gt1=np.diag(cdist(test,train))
D_l2_gt2=np.diag(cdist(train,test))

D_cos_pred1=pairwise_distance(test,train,'cosine')
D_cos_pred2=pairwise_distance(train,test,'cosine')

D_cos_gt1=np.diag(cdist(test,train,'cosine'))
D_cos_gt2=np.diag(cdist(train,test,'cosine'))

print(D_l2_pred1.shape)
print(D_l2_pred2.shape)
print(D_cos_pred1.shape)
print(D_cos_pred2.shape)



print(torch.max(torch.abs(D_l2_pred1-D_l2_gt1)))
print(torch.max(torch.abs(D_l2_pred2-D_l2_gt2)))

print(torch.max(torch.abs(D_cos_pred1-D_cos_gt1)))
print(torch.max(torch.abs(D_cos_pred2-D_cos_gt2)))
