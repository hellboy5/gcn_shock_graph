clear all
close all

files=textread('tign_train.txt','%s\n');

idx=randi(length(files));

zip_file=files{idx};

[path,name,ext]=fileparts(zip_file);

obj=strrep(name,'_se_tcg','.JPEG');

fname=[path '/' obj];
copyfile(fname)
unzip(zip_file)
plot_graph(obj);

eval(['delete ' obj])
eval(['delete ' name '*'])