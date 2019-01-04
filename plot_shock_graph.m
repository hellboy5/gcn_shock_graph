function []=plot_shock_graph(shock_samples,shock_paths,colort)

%figure
% fields=fieldnames(shock_nodes);
% 
% for k=1:length(fields)
%     n1=getfield(shock_nodes,fields{k});
%     for ss=1:length(n1.neighbor)
%         n2=getfield(shock_nodes,['node' num2str(n1.neighbor(ss))]);
%         xcoord=[n1.point(1) n2.point(1)];
%         ycoord=[n1.point(2) n2.point(2)];
%         plot(xcoord+1,ycoord+1,'g','LineWidth',3)
%     end
% end

for k=1:length(shock_paths)
   path=shock_paths{k};
   
   index1 = find(shock_samples(:,1)==path(1));
   point1=shock_samples(index1,2:3);
   temp=shock_samples(index1,4:end);
   
   coords=[point1];
   bp1=[temp];
   
   for p=2:length(path)
       
      
      index2 = find(shock_samples(:,1)==path(p));
      point2=shock_samples(index2,2:3);
      temp=shock_samples(index2,4:end);
      
      coords=[coords ; point2];
      bp2=[temp];
   end
    %plot([coords(1,1)+1 bp1(1)+1],[coords(1,2)+1 bp1(2)+1],'b','LineWidth',3)
    %plot([coords(end,1)+1 bp2(1)+1],[coords(end,2)+1 bp2(2)+1],'b','LineWidth',3)
   
    plot(coords(:,1)+1,coords(:,2)+1,colort,'LineWidth',3)
    plot(coords(1,1)+1,coords(1,2)+1,['.' colort],'MarkerSize',20)
    plot(coords(end,1)+1,coords(end,2)+1,['.' colort],'MarkerSize',20)
%     plot(coords(:,1)+1,coords(:,2)+1,'g','LineWidth',3)
%     plot(coords(1,1)+1,coords(1,2)+1,'g.','MarkerSize',5)
    
    
    hold on
end

