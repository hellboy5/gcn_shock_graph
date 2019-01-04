clear all
close all

I=imread('White_Eyed_Vireo_0015_159081.png');
[shock_samples,shock_edges]=read_shock_file('White_Eyed_Vireo_0015_159081.esf');
F=textread('White_Eyed_Vireo_0015_159081_feature.txt');
A=textread('White_Eyed_Vireo_0015_159081_adj_matrix.txt');
G=digraph(A);

xdata=(F(:,2)*size(I,2))+1;
ydata=(F(:,1)*size(I,1))+1;
imshow(I)
hold on
plot_shock_graph(shock_samples,shock_edges,'g')
plot(G,'Xdata',xdata,'Ydata',ydata)

for s=1:size(F,1)
    
%     pt=[F(s,1) F(s,2)];
%     % loop over theta
%    for t=4:6
%         
%         ang=F(s,t);
%         
%         if ang > 0
%             dx=cos(ang);
%             dy=sin(ang);
%             quiver(pt(1)+1,pt(2)+1,dx,dy,10,'Color','c');
%         end
%     end
%     
%     for t=7:9
%         
%         ang=F(s,t);
%         
%         if ang > 0
%             dx=cos(ang);
%             dy=sin(ang);
%             quiver(pt(1)+1,pt(2)+1,dx,dy,10,'Color','m');
%         end
%     end
%     
    
    
     for t=10:2:15
        
        bnd_pt=[F(s,t+1)*size(I,2) F(s,t)*size(I,1)]+1;
        
        plot(bnd_pt(1),bnd_pt(2),'r.','MarkerSize',20)
       
    end
    
end