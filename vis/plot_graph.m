function []=plot_graph(image_name)

[path,name,ext]=fileparts(image_name);


% Read image
I=imread([path '/' name '.png']);

%Find esf file
esf_files=dir([path '/' name '*.esf']);

%Find h5 file
h5_files=dir([path '/' name '*.h5']);



%Read esf file
[shock_samples,shock_edges]=read_shock_file([ esf_files(1).folder '/' esf_files(1).name]);
sg_file=[h5_files(1).folder '/' h5_files(1).name];
F = h5read(sg_file,'/feature').';
A = h5read(sg_file,'/adj_matrix').';
debug=h5read(sg_file,'/debug');


cons=read_cemv_file([path '/' name  '_to_msel_200_1-5_1.cemv']);

ref_pt=debug(1:2);
max_offsets=debug(3:4);
max_radius=debug(5);

G=digraph(A);

xdata=((F(:,2)*max_offsets(2))+ref_pt(2))+1;
ydata=((F(:,1)*max_offsets(1))+ref_pt(1))+1;


imshow(I)
hold on
plot_shock_graph(shock_samples,shock_edges,'g')
draw_contours(cons,0,0,'c');
plot(G,'Xdata',xdata,'Ydata',ydata)

for s=1:size(F,1)
    
    %      pt=[xdata(s) ydata(s)];
    %      rad=F(s,3)*max_radius;
    %      viscircles(pt,rad,'Color','m');
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
        

        
        if F(s,t) ~= 0 && F(s,t+1) ~= 0
            conx=((F(s,t+1)*max_offsets(2))+ref_pt(2))+1;
            cony=((F(s,t)*max_offsets(1))+ref_pt(1))+1;
            plot(conx,cony,'r.','MarkerSize',20)
        end
        
    end
    
end