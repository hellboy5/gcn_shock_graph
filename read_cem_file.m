function [contours]=read_cem_file(file)

lines={};
fid=fopen(file);
count=1;
while 1
    tline = fgetl(fid);
    if ~ischar(tline)
        break
    end
    lines{count}=tline;
    count=count+1;
end
fclose(fid);

count=1;
contours={};

for k=1:length(lines)
   if ( strcmp(lines{k},'[BEGIN CONTOUR]'))
       
       edge_count_line = lines{k+1};
       [token, remain] = strtok(edge_count_line,'=');
       edge_count = str2num(remain(2:end));
       start=k+2;
       stop=k+1+edge_count;
       con=zeros(edge_count,2);
       index=1;
       for s=start:stop
          point=lines{s};
          index1=max(strfind(point,'['));
          index2=max(strfind(point,']'));
          [x,y]=strread(point(index1+1:index2-1),'%f %f','delimiter',',');
          con(index,:)=[ x y];
          index=index+1;
       end
       
       contours{count}=con;
       count=count+1;
       
       
   end
    
end



