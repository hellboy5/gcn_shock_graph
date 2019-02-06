clear all
close all

unit_test_dir='/Users/naraym1/work/cifar_100/unit_train_dir';

files=dir([unit_test_dir '/*.h5']);

numb_filters=2;

nodes=294;
gt_batch=zeros(numb_filters*nodes,nodes,100);

for f=1:length(files)
    fname=[files(f).folder '/' files(f).name];
    F = h5read(fname,'/feature').';
    A = h5read(fname,'/adj_matrix').';
    diff=nodes-size(A,1);
    pad_A=padarray(A,[diff diff],0,'post');
    pad_A=pad_A+eye(nodes);
    D=diag(sum(pad_A,2).^(-1/2));
    norm_adj=D*pad_A*D;
    if ( numb_filters == 2)
       norm_adj=[eye(nodes) ; norm_adj]; 
    end
    gt_batch(:,:,f)=norm_adj;
end


test_batch=h5read('test.h5','/batch');
test_labels=h5read('test.h5','/labels');

max_diff=zeros(size(gt_batch,3),1);
for z=1:size(gt_batch,3)
   max_diff(z)=max(max(abs(gt_batch(:,:,z).'-test_batch(:,:,z))));
   

    
end

isequal(test_labels,eye(length(files)))