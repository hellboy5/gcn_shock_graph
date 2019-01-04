
function [shock_samples,shock_edges]=read_shock_file(file)


lines={};
fid=fopen(file);
count=1;
while 1
    tline = fgetl(fid);
    if ~ischar(tline)
        break
    end
    if ( strcmp(tline,'Begin [EDGE DESCRIPTION]'))
        start=count+2;
    elseif ( strcmp(tline,'End [EDGE DESCRIPTION]'))
        stop=count-1;
    end
    lines{count}=tline;
    count=count+1;
end
fclose(fid);

shock_samples=[];
shock_edges={};

% shock_nodes=struct;
% 
% data_points=[];

for k=1:length(lines)
    
    if ( strcmp(lines{k},'Begin SAMPLE'))
        
        sample_id_line = lines{k+1};
        [token, remain] = strtok(sample_id_line,' ');
        sample_id =str2num(remain);
        
        point_line = lines{k+2};
        index1=strfind(point_line,'(');
        index2=strfind(point_line,')');
        point_string=point_line(max(index1)+1:max(index2)-1);
        [x,y,t]=strread(point_string,'%f%f%f','delimiter',',');
        
        data=[sample_id x y ];
        
        boundary_point_line = lines{k+8};
        ind1=strfind(boundary_point_line,'(');
        ind2=strfind(boundary_point_line,')');
        bp1=boundary_point_line(min(ind1)+1:min(ind2)-1);
        bp2=boundary_point_line(max(ind1)+1:max(ind2)-1);
        bp1=['[' bp1 ']'];
        bp2=['[' bp2 ']'];
        
        bp1_n=str2num(bp1);
        bp2_n=str2num(bp2);
       
        data=[sample_id x y bp1_n bp2_n];
        shock_samples=[shock_samples ; data];
        
        
        
        
    end
    
end

for k=start:stop
    edge_line=lines{k};
    ind1=strfind(edge_line,'[');
    ind2=strfind(edge_line,']');
    path=edge_line(max(ind1):max(ind2));
    
    path=str2num(path);
    shock_edges=[shock_edges ; path];
    
end
    
% for k=start:stop
%    
%    line=lines{k};
%    ind1=strfind(line,'[');
%    ind2=strfind(line,']');
%    cw_node=line(min(ind1):min(ind2));
%    sample_points=line(max(ind1):max(ind2));
%    [node_id,remain] = strtok(line,' ');
%    
%    sample_points=[str2num(sample_points)];
%    node_id=str2num(node_id);
%    cw_node=[str2num(cw_node)];
%    
%    
%    str=['shock_nodes.node' num2str(node_id) '.neighbor= [' num2str(cw_node) '];'];
%    eval(str)
%    
%    for c=1:length(cw_node)
%        index = find(shocks(:,1)==sample_points(c));
%        point=shocks(index,2:end);
%        str=['shock_nodes.node' num2str(cw_node(c)) '.point=[' num2str(point) '];'];
%        eval(str)
%        
%    end
%  
%   
% end

