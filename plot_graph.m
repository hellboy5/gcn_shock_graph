function []=plot_graph(image_name)

[path,name,ext]=fileparts(image_name);


% Read image
I=imread([path '/' name '.png']);

%Read esf file
[shock_samples,shock_edges]=read_shock_file([ path '/' name '.esf']);
F=textread([path '/' name  '_feature.txt']);
A=textread([path '/' name  '_adj_matrix.txt']);
debug=textread([path '/' name  '_debug.txt']);
cons=read_cem_file([path '/' name  '_to_msel_200_1-5_1.cemv']);

ref_pt=debug(1:2);
max_offsets=debug(3:4);
max_radius=debug(5);

G=digraph(A);

xdata=((F(:,2)*max_offsets(2))+ref_pt(2))+1;
ydata=((F(:,1)*max_offsets(1))+ref_pt(1))+1;


%imshow(I)
hold on
plot_shock_graph(shock_samples,shock_edges,'g')
draw_contours(cons,0,0,'c');
plot(G,'Xdata',xdata,'Ydata',ydata)

for s=1:size(F,1)

     pt=[xdata(s) ydata(s)];
     rad=F(s,3)*max_radius;
     viscircles(pt,rad,'Color','m');
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
        
        conx=((F(s,t+1)*max_offsets(2))+ref_pt(2))+1;
        cony=((F(s,t)*max_offsets(1))+ref_pt(1))+1;

        
        
        plot(conx,cony,'r.','MarkerSize',20)
       
    end
    
end