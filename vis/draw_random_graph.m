clear all
close all


files=dir('/Users/naraym1/work/cifar_100/train/*png');
idx=randi(length(files),1);

fname=[files(idx).folder '/' files(idx).name ];

plot_graph(fname);