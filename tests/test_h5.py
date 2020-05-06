import glob
import os
import h5py
import numpy as np

def read_orig_data(sg_file):
    fid=h5py.File(sg_file,'r')
    
    # read in features
    feature_data=fid.get('feature')
    F_matrix=np.array(feature_data).astype(np.float32)
    sg_features=F_matrix[:,:58]
    color_features=F_matrix[:,118:139]

    fid.close()

    return sg_features,color_features

def read_new_data(sg_file):
    fid=h5py.File(sg_file,'r')

    # read in features
    feature_data=fid.get('feature')
    temp=np.array(feature_data).astype(np.float32)
    sg_features=temp[:,:67]
    sg_features=np.delete(sg_features,[38,39,40,51,52,53,64,65,66],axis=1)
    color_features=temp[:,127:]

    fid.close()

    return sg_features,color_features
    
new='/home/naraym1/stl10/se_tcg_train_fc'
old='/home/naraym1/stl10/se_tcg_train_allf'

new_files=glob.glob(new+'/*h5')

data=np.array([]).reshape(0,3)

for idx in range(len(new_files)):
    print('Working on ',idx)
    fname=new_files[idx]
    suffix=os.path.basename(fname)
    orig_fname=old+'/'+suffix
    if os.path.exists(orig_fname):
        orig_F,orig_C=read_orig_data(orig_fname)
        new_F,new_C=read_new_data(fname)
        aa=np.max(np.abs(orig_F[:,:28]-new_F[:,:28]))
        bb=np.max(np.abs(orig_F-new_F))
        cc=np.max(np.abs(orig_C-new_C))

        dd=np.array([aa,bb,cc]).reshape(1,3)
        data=np.concatenate((data,dd),axis=0)
        
        if aa>0:
            np.savetxt('orig_bad.txt',orig_F,delimiter=' ')
            np.savetxt('new_bad.txt',new_F,delimiter=' ')
            print(aa)
            exit()
print(data)
np.savetxt('foobar.txt',data,delimiter=' ')
print(np.mean(data,axis=0))
