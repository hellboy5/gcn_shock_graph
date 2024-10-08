import re
import sys
import os
import cv2
import numpy as np
import h5py
import math
import networkx as nx
import argparse
#import matplotlib.pyplot as plt
import copy
from shape_context import SC
from collections import defaultdict
from collections import namedtuple
from math import acos
from math import pi
from math import cos
from math import sin
from operator import itemgetter
from scipy.spatial.distance import cdist
from scipy import ndimage

node_order=dict()
order_mapping=dict()
node_mapping=dict()
edge_samples=dict()
edge_to_samples=dict()
curve_stats=dict()
samp_to_node_mapping=defaultdict(list)
edge_mapping=defaultdict(list)
adj_nodes_mapping=defaultdict(list)
Sample=namedtuple('Sample','id pt type radius theta speed phi plus_pt \
minus_pt plus_theta minus_theta')
CurveProps=namedtuple('CurveProps','SCurve SLength SAngle PCurve PLength PAngle MCurve MLength MAngle PolyArea AvgColor HistColor')
highest_degree=0
special_2_nodes=set()
revisit_nodes=set()
ZERO_TOLERANCE=1E-1


def get_histogram(coords,color_space,numb_bins=21):
    if len(color_space.shape) == 2:
        print('gray scale converting to color')
        color_space=np.repeat(color_space[:,:,np.newaxis],3,axis=2)

    samples=[coords[:,1],coords[:,0]]
    
    chan1_values  = ndimage.map_coordinates(color_space[:,:,2],samples,mode='nearest')
    chan2_values  = ndimage.map_coordinates(color_space[:,:,1],samples,mode='nearest')
    chan3_values  = ndimage.map_coordinates(color_space[:,:,0],samples,mode='nearest')

    chan1_hist,_ = np.histogram(chan1_values,bins=numb_bins,range=(0.0,255.0))
    chan2_hist,_ = np.histogram(chan2_values,bins=numb_bins,range=(0.0,255.0))
    chan3_hist,_ = np.histogram(chan3_values,bins=numb_bins,range=(0.0,255.0))
    
    return (chan1_hist,chan2_hist,chan3_hist)
    
def get_frag_average_color(coords,color_space):

    if len(color_space.shape) == 2:
        print('gray scale converting to color')
        color_space=np.repeat(color_space[:,:,np.newaxis],3,axis=2)

    samples=[coords[:,1],coords[:,0]]
    
    chan1_values  = ndimage.map_coordinates(color_space[:,:,2],samples,mode='nearest')
    chan2_values  = ndimage.map_coordinates(color_space[:,:,1],samples,mode='nearest')
    chan3_values  = ndimage.map_coordinates(color_space[:,:,0],samples,mode='nearest')

    chan1_values = np.mean(chan1_values)
    chan2_values = np.mean(chan2_values)
    chan3_values = np.mean(chan3_values)
    
    return np.array([chan1_values,chan2_values,chan3_values])


def get_pixel_image(coords,color_space):
    
    if len(color_space.shape) == 2:
        print('gray scale converting to color')
        color_space=np.repeat(color_space[:,:,np.newaxis],3,axis=2)

    samples=[coords[:,1],coords[:,0]]

    output=np.zeros((1,3))
    output[0,0] = ndimage.map_coordinates(color_space[:,:,2],samples,mode='nearest')[0]
    output[0,1] = ndimage.map_coordinates(color_space[:,:,1],samples,mode='nearest')[0]
    output[0,2] = ndimage.map_coordinates(color_space[:,:,0],samples,mode='nearest')[0]

    return output

def get_pixel_values(F_matrix,color_space):

    if len(color_space.shape) == 2:
        print('gray scale converting to color')
        color_space=np.repeat(color_space[:,:,np.newaxis],3,axis=2)

    pixel_values=np.zeros((F_matrix.shape[0],21),dtype=np.float32)
    
    zero_set=np.array([0.0,0.0])
    coords= np.zeros((1,2))
    for f in range(F_matrix.shape[0]):
        coords[0,0]=F_matrix[f,0]
        coords[0,1]=F_matrix[f,1]
        sg_cs=get_pixel_image(coords,color_space)
        pixel_values[f,:3]=sg_cs

        # samples = np.empty(shape=(0,2))
        # colors  = np.empty(shape=(0,3))

        # samples=np.vstack((samples,coords))
        # colors=np.vstack((colors,pixel_values[f,:3]))
        
        coords[0,0]=F_matrix[f,9]
        coords[0,1]=F_matrix[f,10]
        bp1_cs=get_pixel_image(coords,color_space)
        pixel_values[f,3:6]=bp1_cs

        # samples=np.vstack((samples,coords))
        # colors=np.vstack((colors,pixel_values[f,3:6]))
        
        coords[0,0]=F_matrix[f,15]
        coords[0,1]=F_matrix[f,16]
        bp1_cs=get_pixel_image(coords,color_space)
        pixel_values[f,12:15]=bp1_cs

        # samples=np.vstack((samples,coords))
        # colors=np.vstack((colors,pixel_values[f,12:15]))
                
        if np.array_equal(F_matrix[f,11:13],zero_set)==False:
            coords[0,0]=F_matrix[f,11]
            coords[0,1]=F_matrix[f,12]
            bp2_cs=get_pixel_image(coords,color_space)
            pixel_values[f,6:9]=bp2_cs

            # samples=np.vstack((samples,coords))
            # colors=np.vstack((colors,pixel_values[f,6:9]))
                            
            coords[0,0]=F_matrix[f,17]
            coords[0,1]=F_matrix[f,18]
            bp2_cs=get_pixel_image(coords,color_space)
            pixel_values[f,15:18]=bp2_cs

            # samples=np.vstack((samples,coords))
            # colors=np.vstack((colors,pixel_values[f,15:18]))
                        
        if np.array_equal(F_matrix[f,13:15],zero_set)==False:
            coords[0,0]=F_matrix[f,13]
            coords[0,1]=F_matrix[f,14]
            bp3_cs=get_pixel_image(coords,color_space)
            pixel_values[f,9:12]=bp3_cs

            # samples=np.vstack((samples,coords))
            # colors=np.vstack((colors,pixel_values[f,9:12]))
            
            coords[0,0]=F_matrix[f,19]
            coords[0,1]=F_matrix[f,20]
            bp3_cs=get_pixel_image(coords,color_space)
            pixel_values[f,18:21]=bp3_cs

            # samples=np.vstack((samples,coords))
            # colors=np.vstack((colors,pixel_values[f,18:21]))

        # np.savetxt(str(f)+'_samples.txt',samples)
        # np.savetxt(str(f)+'_hist.txt',colors)
        
    return pixel_values


def read_cemv_file(fid):

    con_points=[]
    file=open(fid,'r')
    lines= file.read().splitlines() 
    start = [s for s,e in enumerate(lines) if e == '[BEGIN CONTOUR]']
    for c in start:
        numb_points=int(lines[c+1].split('=')[1])
        start=c+2
        for idx in range(start,start+numb_points):
            b_start=lines[idx].rfind('[')
            b_stop=lines[idx].rfind(']')
            point=lines[idx][b_start+1:b_stop].split(',')
            x=float(point[0])
            y=float(point[1])
            con_points.append((x,y))


    file.close()
    return con_points



    
def coarsen_graph(paths):


    adj_nodes_mapping.clear()
    edge_mapping.clear()
    node_order.clear()
    
    for vals in paths.values():
        source_id=order_mapping[vals[0]]
        target_id=order_mapping[vals[-1]]

        source_rad=node_mapping[source_id].radius[0]
        target_rad=node_mapping[target_id].radius[0]


        if source_rad <= target_rad:
            edge_mapping[source_id].append(target_id)
        else:
            edge_mapping[target_id].append(source_id)

        adj_nodes_mapping[source_id].append(target_id)
        adj_nodes_mapping[target_id].append(source_id)


        for id in range(1,len(vals)-1):
            sample_id=order_mapping[vals[id]]
            del node_mapping[sample_id]

    

    compute_sorted_order()
    
def translate_points(pt,v,time):

    trans_points=np.zeros((pt.shape[0],2),dtype=np.float64)
    x=pt[:,0]+time*np.cos(v)
    y=pt[:,1]+time*np.sin(v)
    trans_points[:,0]=x
    trans_points[:,1]=y

    return trans_points

def angleDiff_new(a1,a2):

    a1=fixAngleMPiPi_new(a1)
    a2=fixAngleMPiPi_new(a2)

    if a1 > a2:
        if (a1-a2) > pi:
            return a1-a2-2*pi
        else:
            return a1-a2
    elif a2 > a1:
        if (a1-a2) < -pi:
            return a1-a2+2*pi
        else:
            return a1-a2

    return 0.0

def l2_dist(pt1,pt2):
    dx   = pt1[0]-pt2[0]
    dy   = pt1[1]-pt2[1]
    ds = math.sqrt(dx*dx + dy*dy)
    return ds

def subsample(sh_pt_,time_,theta_,phi_,bdry_plus_,bdry_minus_,subsample_ds=5.0):

  sub_sh_pt=[]
  sub_time=[]
  sub_theta=[]
  sub_phi=[]
  sub_bdry_plus=[]
  sub_bdry_minus=[]
  
  #keep the first and last points and subsample the rest

  
  #first sample
  sub_sh_pt.append(sh_pt_[0])
  sub_time.append(time_[0])  
  sub_theta.append(theta_[0])
  sub_phi.append(phi_[0])
  sub_bdry_plus.append(bdry_plus_[0])
  sub_bdry_minus.append(bdry_minus_[0])

  So = sh_pt_[0]
  Se = sh_pt_[-1]
  PBo = bdry_plus_[0]
  PBe = bdry_plus_[-1]
  MBo = bdry_minus_[0]
  MBe = bdry_minus_[-1]

  for i in range(1,len(sh_pt_)-1):
      Sn=sh_pt_[i]
      PBn=bdry_plus_[i]
      MBn=bdry_minus_[i]

      #core subsampling criteria
      if ( (l2_dist(So,Sn)  > subsample_ds) and (l2_dist(Sn,Se)  >subsample_ds) ) or \
         ( (l2_dist(PBo,PBn)> subsample_ds) and (l2_dist(PBn,PBe)>subsample_ds) ) or \
         ( (l2_dist(MBo,MBn)> subsample_ds) and (l2_dist(MBn,MBe)>subsample_ds) ):
    
          sub_sh_pt.append(sh_pt_[i])
          sub_time.append(time_[i])
          sub_theta.append(theta_[i])
          sub_phi.append(phi_[i])      
          sub_bdry_plus.append(bdry_plus_[i])
          sub_bdry_minus.append(bdry_minus_[i])

          So=Sn
          PBo=PBn
          MBo=MBn
      
  #last sample
  sub_sh_pt.append(sh_pt_[-1])
  sub_time.append(time_[-1])
  sub_theta.append(theta_[-1])
  sub_phi.append(phi_[-1])
  sub_bdry_plus.append(bdry_plus_[-1])
  sub_bdry_minus.append(bdry_minus_[-1])
  
  return sub_sh_pt,sub_theta,sub_phi,sub_time,sub_bdry_plus,sub_bdry_minus

# This function takes a sampled shock curve and interpolates it.
def interpolate(sh_pt,time,theta,phi,interpolate_ds=1.0):

    # we need to interpolate along the length of the shock curve
    # to fill the gaps between the shock samples
  
    # add the very first sample
    sh_pt_=[]
    time_=[]
    theta_=[]
    phi_=[]

    sh_pt_.append(sh_pt[0])
    time_.append(time[0])
    theta_.append(theta[0])
    phi_.append(phi[0])


    for i in range(1,len(sh_pt)):

        dphi = angleDiff_new(phi[i],phi[i-1])
        dtheta = angleDiff_new(theta[i], theta[i-1])
        dx = sh_pt[i][0]-sh_pt[i-1][0]
        dy = sh_pt[i][1]-sh_pt[i-1][1]
        ds = math.sqrt(dx*dx + dy*dy)
        dt = time[i]-time[i-1]
        apprxds = ds+(abs(dtheta)+abs(dphi))*(time[i-1]+time[i])/2.0

        # (this is Amir's working interpolation as well.)
        if apprxds>interpolate_ds:

            num = int(apprxds/interpolate_ds)

            for j in range(1,num):

                ratio = float(j)/float(num)

                p_int=(sh_pt[i-1][0]+ratio*dx,sh_pt[i-1][1]+ratio*dy)
                time_int = time[i-1] + ratio*dt
                phi_int = phi[i-1] + ratio*dphi
                theta_int = theta[i-1] + ratio*dtheta

                sh_pt_.append(p_int)
                time_.append(time_int)
                theta_.append(theta_int)
                phi_.append(phi_int)
                

        # add the current original sample
        sh_pt_.append(sh_pt[i])
        time_.append(time[i])
        theta_.append(theta[i])
        phi_.append(phi[i])

    return sh_pt_,theta_,phi_,time_


def get_degree(G,node):
    gg=order_mapping[node]
    if str(gg) in revisit_nodes:
        return G.degree[node]+1
    else:
        return G.degree[node]
    
def check_paths(G,paths):

    nodes=G.nodes
    hist=defaultdict(int)
    for val in paths.values():
        for vertices in val:
            hist[vertices]+=1

    flag=True
    for nn in list(G.nodes):
        truth=get_degree(G,nn)
        if truth==2:
            truth-=1

        if nn not in hist:
            print('Nodes not visited')
            flag=False
            break
        
        counts=hist[nn]
        if (counts != truth) and (nn not in special_2_nodes):
            print('Nodes not visited enough or too many times')
            flag=False
            break


    return flag
        
def get_paths(A,color_space):
    
    G=nx.from_numpy_matrix(A)
    components_visited=set()
    degree_three_one_nodes=[]
    visited=set()
    for ids in list(G.nodes):
        if get_degree(G,ids) >= 3 or get_degree(G,ids)==1:
            degree_three_one_nodes.append(ids)
            for x,c in enumerate(nx.connected_components(G)):
                if ids in c:
                    components_visited.add(x)
                
    if len(components_visited) != nx.number_connected_components(G):
        print('We have not visited all connected components')
        for x,c in enumerate(nx.connected_components(G)):
            if x not in components_visited:
                degree_three_one_nodes.append(c.pop())
        
    all_paths=dict()
    
    for idx in degree_three_one_nodes:

        visited.add(idx)
        for vx in G.neighbors(idx):
            path=[]
            path.append(idx)

            if vx not in visited:
                path.append(vx)
                pairs,visited=path_dfs(G,path,visited)
                all_paths[pairs[0]]=pairs[1]

    curve_stats.clear()
    #fid=open('curves.txt','w')
    for vals in all_paths.values():

        source_id=order_mapping[vals[0]]
        shock_curve=[]
        radius=[]
        theta=[]
        phi=[]
        time=[]
        overall_key=(str(order_mapping[vals[0]]),str(order_mapping[vals[-1]]))
        
        for idx in range(1,len(vals)):
            target_id=order_mapping[vals[idx]]
            key=(str(source_id),str(target_id))
            
            flip=False
            if key in edge_to_samples:

                value=edge_to_samples[key]
                start_idx=node_mapping[int(key[0])].id.index(int(value[0]))
                stop_idx=node_mapping[int(key[1])].id.index(int(value[-1]))

                shock_start=node_mapping[int(key[0])].pt[start_idx]
                shock_stop=node_mapping[int(key[1])].pt[stop_idx]

                radius_start=node_mapping[int(key[0])].radius[start_idx]
                radius_stop=node_mapping[int(key[1])].radius[stop_idx]

                theta_start=node_mapping[int(key[0])].theta[start_idx]
                theta_stop=node_mapping[int(key[1])].theta[stop_idx]

                phi_start=node_mapping[int(key[0])].phi[start_idx]
                phi_stop=node_mapping[int(key[1])].phi[stop_idx]

            else:
                key=(str(target_id),str(source_id))
                flip=True
                value=edge_to_samples[key]

                start_idx=node_mapping[int(key[1])].id.index(int(value[-1]))
                stop_idx=node_mapping[int(key[0])].id.index(int(value[0]))

                shock_start=node_mapping[int(key[1])].pt[start_idx]
                shock_stop=node_mapping[int(key[0])].pt[stop_idx]

                radius_start=node_mapping[int(key[1])].radius[start_idx]
                radius_stop=node_mapping[int(key[0])].radius[stop_idx]

                theta_start=angle0To2Pi(node_mapping[int(key[1])].theta[start_idx]+pi)
                theta_stop=angle0To2Pi(node_mapping[int(key[0])].theta[stop_idx]+pi)

                phi_start=speedToPhi(node_mapping[int(key[1])].speed[start_idx],pi)
                phi_stop=speedToPhi(node_mapping[int(key[0])].speed[stop_idx],pi)

            shock_curve.append(shock_start)
            radius.append(radius_start)
            theta.append(theta_start)
            phi.append(phi_start)

            if flip:
                shock_curve.extend([edge_samples.get(int(value[id])).pt for id in range(-2,len(value)*-1,-1)])
                radius.extend([edge_samples.get(int(value[id])).radius for id in range(-2,len(value)*-1,-1)])                
                theta.extend([angle0To2Pi(edge_samples.get(int(value[id])).theta+pi) for id in range(-2,len(value)*-1,-1)])
                phi.extend([speedToPhi(edge_samples.get(int(value[id])).speed,pi) for id in range(-2,len(value)*-1,-1)])
            else:
                shock_curve.extend([edge_samples.get(int(value[id])).pt for id in range(1,len(value)-1)])
                radius.extend([edge_samples.get(int(value[id])).radius for id in range(1,len(value)-1)])
                theta.extend([edge_samples.get(int(value[id])).theta for id in range(1,len(value)-1)])
                phi.extend([edge_samples.get(int(value[id])).phi for id in range(1,len(value)-1)])
                
           
           
            shock_curve.append(shock_stop)
            radius.append(radius_stop)
            theta.append(theta_stop)
            phi.append(phi_stop)

            source_id=target_id
            

        if len(theta)==0 or len(phi)==0:
            print(plus_curve)
            print(vals)
            print('very very bad')

        shock_curve,theta,phi,radius=interpolate(shock_curve,radius,theta,phi)
        
        plus_angles=np.array(theta,dtype=np.float64)+np.array(phi,dtype=np.float64)
        minus_angles=np.array(theta,dtype=np.float64)-np.array(phi,dtype=np.float64)

        plus_curve=translate_points(np.array(shock_curve,dtype=np.float64),plus_angles,radius)
        minus_curve=translate_points(np.array(shock_curve,dtype=np.float64),minus_angles,radius)
      
        shock_curve,theta,phi,radius,plus_curve,minus_curve=subsample(shock_curve,radius,theta,phi,plus_curve,minus_curve)
        
        shock_length,shock_totalCurvature,shock_angle=computeCurveStats(shock_curve)
        plus_length,plus_totalCurvature,plus_angle=computeCurveStats(plus_curve)
        minus_length,minus_totalCurvature,minus_angle=computeCurveStats(minus_curve)

        points=sample_fragment(shock_curve,theta,phi,radius)

        avg_color=get_frag_average_color(points,color_space)

        if len(plus_curve)==0:
            plus_length         = shock_length
            plus_totalCurvature = shock_totalCurvature
            plus_angle          = shock_angle

        if len(minus_curve)==0:
            minus_length         = shock_length
            minus_totalCurvature = shock_totalCurvature
            minus_angle          = shock_angle

        area=0.0
        xpoly=[]
        ypoly=[]
        if len(plus_curve) and len(minus_curve):
            poly=[]
            poly.append(shock_curve[0])
            poly.extend(plus_curve)
            poly.append(shock_curve[-1])
            minus_curve=np.flip(minus_curve,0)
            poly.extend(minus_curve)
            poly.append(shock_curve[0])
            totals=list(zip(*poly))
            xpoly=totals[0]
            ypoly=totals[1]
            area=polyArea(np.array(totals[0]),np.array(totals[1]))


        if len(plus_curve) and len(minus_curve):
            if plus_curve[0][0] > minus_curve[0][0]:
                plus_totalCurvature,minus_totalCurvature=minus_totalCurvature,plus_totalCurvature
                plus_angle,minus_angle=minus_angle,plus_angle
                plus_length,minus_length=minus_length,plus_length
                  
        stats=CurveProps(SCurve=shock_totalCurvature,
                         SLength=shock_length,
                         SAngle=shock_angle,
                         PCurve=plus_totalCurvature,
                         PLength=plus_length,
                         PAngle=plus_angle,
                         MCurve=minus_totalCurvature,
                         MLength=minus_length,
                         MAngle=minus_angle,
                         PolyArea=area,
                         AvgColor=avg_color)
          
        curve_stats[overall_key]=stats


     #    fid.write("%i "% len(shock_curve))
    #     for wt in range(len(shock_curve)):

    #         if wt == len(shock_curve)-1:
    #             fid.write("%6.16f %6.16f\n" % (shock_curve[wt][0],shock_curve[wt][1]))
    #         else:
    #             fid.write("%6.16f %6.16f " % (shock_curve[wt][0],shock_curve[wt][1]))

    #     fid.write("%i "% len(theta))

    #     if len(theta)==0:
    #         fid.write("\n")
            
    #     for wt in range(len(theta)):

    #         if wt == len(theta)-1:
    #             fid.write("%6.16f\n" % (theta[wt]))
    #         else:
    #             fid.write("%6.16f " % (theta[wt]))


    #     fid.write("%i "% len(phi))

    #     if len(phi)==0:
    #         fid.write("\n")

    #     for wt in range(len(phi)):
                    
    #         if wt == len(phi)-1:
    #             fid.write("%6.16f\n" % (phi[wt]))
    #         else:
    #             fid.write("%6.16f " % (phi[wt]))

    #     fid.write("%i "% len(plus_curve))
    #     for wt in range(len(plus_curve)):

    #         if wt == len(plus_curve)-1:
    #             fid.write("%6.16f %6.16f\n" % (plus_curve[wt][0],plus_curve[wt][1]))
    #         else:
    #             fid.write("%6.16f %6.16f " % (plus_curve[wt][0],plus_curve[wt][1]))

    #     fid.write("%i "% len(minus_curve))
    #     for wt in range(len(minus_curve)):

    #         if wt == len(minus_curve)-1:
    #             fid.write("%6.16f %6.16f\n" % (minus_curve[wt][0],minus_curve[wt][1]))
    #         else:
    #             fid.write("%6.16f %6.16f " % (minus_curve[wt][0],minus_curve[wt][1]))


    #     fid.write("%i "% len(xpoly))
    #     fid.write("%6.16f "% area)
        
    #     for wt in range(len(xpoly)):

    #         if wt == len(xpoly)-1:
    #             fid.write("%6.16f %6.16f\n" % (xpoly[wt],ypoly[wt]))
    #         else:
    #             fid.write("%6.16f %6.16f " % (xpoly[wt],ypoly[wt]))


    # fid.close()
    return G,all_paths

def path_dfs(G,path,visited):

    traversal=[path[-1]]
    
    while len(traversal):
        node=traversal.pop(-1)

        if get_degree(G,node)>=3 or get_degree(G,node)==1:
            break
        
        visited.add(node)
        neighbors=G.neighbors(node)
        for val in neighbors:
            if val not in visited:
                path.append(val)
                traversal.append(val)


    #check path ends with a degree three or degree one node
    if get_degree(G,path[-1]) == 2:
        print('Found path that does not end in degree 3 or 1')
        node=path[-1]
        neighbors=G.neighbors(node)
        for val in neighbors:
            if val in visited:
                if get_degree(G,val)>=3 or get_degree(G,val)==1:
                    path.append(val)
                    break

        if get_degree(G,path[-1])==2:
            print('Path is still 2 at end')
            path.append(path[0])
            special_2_nodes.add(path[0])
        
    
    return (str(sorted(path)),path),visited
        
# def vis(G,I,positions,all_paths):

#     pos_flipped=np.fliplr(positions)
#     fig, ax = plt.subplots()
#     plt.imshow(I)
#     nx.draw(G,pos=pos_flipped,node_size=20)

#     #create pos vector
#     pos_vector=dict()
#     for x in range(positions.shape[0]):
#         pos_vector[x]=pos_flipped[x,:]
        
#     for path in all_paths.values():
#         a1=copy.deepcopy(path)
#         a2=copy.deepcopy(path)
#         a1.pop(-1)
#         a2.pop(0)
#         elist=zip(a1,a2)
#         colors=np.random.rand(3)
#         colors=np.tile(colors,(len(elist),1))
        
#         nx.draw_networkx_edges(G,
#                                pos=pos_vector,
#                                edgelist=elist,
#                                edge_color=colors,width=10)
        
    
#     plt.show()
    

def polyArea(x,y):
    # coordinate shift
    x_ = x - x.mean()
    y_ = y - y.mean()
    # everything else is the same as maxb's code
    correction = x_[-1] * y_[0] - y_[-1]* x_[0]
    main_area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
    return 0.5*np.abs(main_area + correction)

def computeDerivatives(curve):

     #Compute derivatives
     dx=[]
     dx.append(0.0)
     dy=[]
     dy.append(0.0)

     px=curve[0][0]
     py=curve[0][1]

     for i in range(1,len(curve)):
          cx=curve[i][0]
          cy=curve[i][1]
          dL=np.hypot(cx-px,cy-py)

          if dL > ZERO_TOLERANCE: 
               dx.append((cx-px)/dL)
               dy.append((cy-py)/dL)
          else:
               dx.append(0.0)
               dy.append(0.0)
    
          px=cx
          py=cy

     return dx,dy

def computeArcLength(curve):

     #Compute arc length and normalized arc length
     arcLength=[]
     s=[]
     length=0
     arcLength.append(0.0)
     s.append(0.0)

     px=curve[0][0]
     py=curve[0][1]

     for i in range(1,len(curve)):          
          cx=curve[i][0]
          cy=curve[i][1]
          dL = np.hypot(cx-px,cy-py)
          length += dL
          arcLength.append(length)
          s.append(dL)
          px=cx
          py=cy

     return arcLength,length

  


def computeCurvatures(dx,dy,arcLength,curveLength):
     
     #Compute curvature
     curvature=[]
     curvature.append(0.0)
     totalCurvature=0.0

     for i in range(1,curveLength):
          pdx=dx[i-1]
          pdy=dy[i-1]
          cdx=dx[i]
          cdy=dy[i]
          dL=arcLength[i]-arcLength[i-1]
          d2x=0
          d2y=0

          if dL > ZERO_TOLERANCE:
               d2x=(cdx-pdx)/dL
               d2y=(cdy-pdy)/dL

          K = 0
          
          if abs(cdx) >= ZERO_TOLERANCE or abs(cdy) >= ZERO_TOLERANCE:
               K=(d2y*cdx-d2x*cdy)/math.pow((math.pow(cdx,2)+math.pow(cdy,2)),3/2)

          curvature.append(K)
          totalCurvature+=K

     return curvature,totalCurvature
  


def computeAngles(curve):
     angle=[]
     angle.append(0.0)
     totalAngleChange=0.0

     px=curve[0][0]
     py=curve[0][1]
     for i in range(1,len(curve)):
          cx=curve[i][0]
          cy=curve[i][1]
          theta=math.atan2(cy-py,cx-px)
          angle.append(theta)
          px=cx
          py=cy
     
     if len(curve) >2 :
          angle[0]=angle[1]
          for i in range(1,len(angle)):
               totalAngleChange += abs(angle[i]-angle[i-1])

     return totalAngleChange
  
def computeCurveStats(curve):

     if len(curve)==0:
          return 0.0,0.0,0.0
     
     arcLength,length=computeArcLength(curve)
     dx,dy=computeDerivatives(curve)
     _,totalCurvature=computeCurvatures(dx,dy,arcLength,len(curve))
     totalAngleChange=computeAngles(curve)

     return length,totalCurvature,totalAngleChange

def sample_node(shock,radius):

    sample_points=np.array([]).reshape(0,2)
    rad_samples=np.linspace(0,1,11)*radius
    angles_samples=(pi/180.0)*np.linspace(0,360,37)

    for rad in rad_samples:
        ray_points=np.zeros((len(angles_samples),2))
        for idx in range(len(angles_samples)):
            theta=angles_samples[idx]
            pt = translatePoint(shock,theta,rad)
            ray_points[idx,0]=pt[0]
            ray_points[idx,1]=pt[1]

        sample_points=np.vstack((sample_points,ray_points))

    return sample_points
        
        
def sample_fragment(shock,theta,phi,radius):

    spacing=[0.0,0.2,0.4,0.6,0.8]
    sample_points=np.array([]).reshape(0,4)
    for idx in range(len(shock)):

        ray_points=np.zeros((len(spacing),4))
        for jdx in range(len(spacing)):
            left_bnd_pt  = translatePoint(shock[idx],theta[idx]+phi[idx], spacing[jdx]*radius[idx])
            right_bnd_pt = translatePoint(shock[idx],theta[idx]-phi[idx], spacing[jdx]*radius[idx])
            ray_points[jdx,0]=left_bnd_pt[0]
            ray_points[jdx,1]=left_bnd_pt[1]
            ray_points[jdx,2]=right_bnd_pt[0]
            ray_points[jdx,3]=right_bnd_pt[1]

        sample_points=np.vstack((sample_points,ray_points))


    points=np.vstack((sample_points[:,:2],sample_points[:,2:4]))
    return points
            

def compute_edge_stats(color_space,numb_bins):

    for key,value in edge_to_samples.items():
          start_idx=node_mapping[int(key[0])].id.index(int(value[0]))
          stop_idx=node_mapping[int(key[1])].id.index(int(value[-1]))

          first_pt=node_mapping[int(key[0])].pt[start_idx]
          shock_curve=[edge_samples.get(int(value[id])).pt for id in range(1,len(value)-1)]
          last_pt=node_mapping[int(key[1])].pt[stop_idx]
          shock_curve.insert(0,first_pt)
          shock_curve.append(last_pt)

          first_theta=node_mapping[int(key[0])].theta[start_idx]
          theta_curve=[edge_samples.get(int(value[id])).theta for id in range(1,len(value)-1)]
          last_theta=node_mapping[int(key[1])].theta[stop_idx]
          theta_curve.insert(0,first_theta)
          theta_curve.append(last_theta)

          first_phi=node_mapping[int(key[0])].phi[start_idx]
          phi_curve=[edge_samples.get(int(value[id])).phi for id in range(1,len(value)-1)]
          last_phi=node_mapping[int(key[1])].phi[stop_idx]
          phi_curve.insert(0,first_phi)
          phi_curve.append(last_phi)

          first_radius=node_mapping[int(key[0])].radius[start_idx]
          radius_curve=[edge_samples.get(int(value[id])).radius for id in range(1,len(value)-1)]
          last_radius=node_mapping[int(key[1])].radius[stop_idx]
          radius_curve.insert(0,first_radius)
          radius_curve.append(last_radius)

          points=sample_fragment(shock_curve,theta_curve,phi_curve,radius_curve)

          avg_color=get_frag_average_color(points,color_space)

          hist_color=tuple()
          
          if numb_bins > 0:
              hist_color=get_histogram(points,color_space,numb_bins)

              # gg=np.vstack((hist_color[0],hist_color[1],hist_color[2]))
              # np.savetxt(str(key[0])+'_s_to_t_'+str(key[1])+'_samples.txt',points)
              # np.savetxt(str(key[0])+'_s_to_t_'+str(key[1])+'_avg.txt',avg_color)
                  
          # np.savetxt(str(key[0])+'_s_to_t_'+str(key[1])+'.txt',points,delimiter=' ')
          # np.savetxt(str(key[0])+'_s_to_t_'+str(key[1])+'_avg.fmt',avg_color,delimiter=' ')
          
          plus_curve=[edge_samples.get(int(value[id])).plus_pt for id in range(1,len(value)-1)]
          minus_curve=[edge_samples.get(int(value[id])).minus_pt for id in range(1,len(value)-1)]

          
          shock_length,shock_totalCurvature,shock_angle=computeCurveStats(shock_curve)
          plus_length,plus_totalCurvature,plus_angle=computeCurveStats(plus_curve)
          minus_length,minus_totalCurvature,minus_angle=computeCurveStats(minus_curve)

          if len(plus_curve)==0:
               plus_length         = shock_length
               plus_totalCurvature = shock_totalCurvature
               plus_angle          = shock_angle

          if len(minus_curve)==0:
               minus_length         = shock_length
               minus_totalCurvature = shock_totalCurvature
               minus_angle          = shock_angle

          area=0.0
          if len(plus_curve) and len(minus_curve):
               poly=[]
               poly.append(shock_curve[0])
               poly.extend(plus_curve)
               poly.append(shock_curve[-1])
               minus_curve.reverse()
               poly.extend(minus_curve)
               poly.append(shock_curve[0])
               totals=list(zip(*poly))
               area=polyArea(np.array(totals[0]),np.array(totals[1]))

        
          # Not sure if this is good revisit
          # if len(plus_curve) and len(minus_curve):
          #     if plus_curve[0][0] > minus_curve[0][0]:
          #         plus_totalCurvature,minus_totalCurvature=minus_totalCurvature,plus_totalCurvature
          #         plus_angle,minus_angle=minus_angle,plus_angle
          #         plus_length,minus_length=minus_length,plus_length

          stats=CurveProps(SCurve=shock_totalCurvature,
                           SLength=shock_length,
                           SAngle=shock_angle,
                           PCurve=plus_totalCurvature,
                           PLength=plus_length,
                           PAngle=plus_angle,
                           MCurve=minus_totalCurvature,
                           MLength=minus_length,
                           MAngle=minus_angle,
                           PolyArea=area,
                           AvgColor=avg_color,
                           HistColor=hist_color)
          
          curve_stats[key]=stats

          
def getLengthSampNode():
     length=0
     for value in samp_to_node_mapping.values():
          length=length+len(value)
     return length

#: Convert an angle to [0, 2Pi) range                                                                   
def angle0To2Pi(angle):

    if angle>=2*pi:
        a=math.fmod(angle,pi*2)
    elif angle < 0:
        a=(2*pi+math.fmod(angle,2*pi))
    else:
        a=angle


    if not (a>=0 and a<2*pi):
        a = 0

    return a

def speedToPhi(speed,offset=0):
     if speed != 0 and speed < 99990:
          phi = offset+acos(-1.0/speed)
          if offset > 0.0:
              phi=offset-acos(-1.0/speed)
     else:
          phi = pi/2.0
     return phi

def fixAngleMPiPi_new(a):
     if a < -pi:
          return a+2.0*pi
     elif a > pi:
          return a-2.0*pi
     else:
          return a

def translatePoint(pt,v,length):
     x = pt[0] + length*cos(v)
     y = pt[1] + length*sin(v)
     return (x,y)


# process node header data
def read_node_header(node_line,lines,numb_nodes):

     global highest_degree
     for ln in range(node_line,node_line+numb_nodes):
          text=lines[ln]
          node_id,label=text.split(' ')[:2]
          samp=Sample(id=[],
                      pt=[],
                      type=label,
                      radius=[],
                      theta=[],speed=[],phi=[],
                      plus_pt=[],minus_pt=[],
                      plus_theta=[], minus_theta=[])
          node_mapping[int(node_id)]=samp
          starts=[match.start() for match in re.finditer(re.escape('['), text)]
          ends=[match.start() for match in re.finditer(re.escape(']'), text)]
          results=list(zip(starts,ends))
          adj_nodes=text[results[0][0]+1:results[0][1]].split(' ')
          adj_samples=text[results[1][0]+1:results[1][1]].split(' ')
          highest_degree=max(highest_degree,len(adj_samples)-1)
          for key in adj_samples:
               if len(key):
                    samp_to_node_mapping[int(key)].append(int(node_id))
          for key in adj_nodes:
               if len(key):
                    adj_nodes_mapping[int(node_id)].append(int(key))

# process node sample data
def read_node_samples(sample_line,lines,sample_data,node_info):

     numb_mappings=getLengthSampNode()
     start=sample_line
     end=sample_line+numb_mappings*sample_data
     for ln in range(start,end,sample_data):
          
          idx=list(range(ln+1,ln+1+node_info))
          
          # get sample id
          text=lines[idx[0]]
          id=int(text.split(' ')[1])

          # get location and radius
          text=lines[idx[1]]
          starts=[match.start() for match in re.finditer(re.escape('('), text)]
          end=[match.start() for match in re.finditer(re.escape(')'), text)]
          results=list(zip(starts,end))
          x,y,radius=text[results[1][0]+1:results[1][1]].split(',')
          pt=(float(x),float(y))
          radius=float(radius)
               
          # get theta
          text=lines[idx[5]]
          theta=np.deg2rad(float(text.split(' ')[1]))

          # get speed , convert to phi
          text=lines[idx[6]]
          speed=float(text.split(' ')[1])
          phi=speedToPhi(speed)

          #get reconstructed boundary points
          left_bnd_pt  = translatePoint(pt,theta+phi, radius)
          right_bnd_pt = translatePoint(pt,theta-phi, radius)

          #get reconstructed boundary tangents
          left_bnd_tangent  = fixAngleMPiPi_new(theta+phi-pi/2.0)
          right_bnd_tangent = fixAngleMPiPi_new(theta-phi+pi/2.0)

          # if left_bnd_pt[0] > right_bnd_pt[0]:
          #     left_bnd_pt,right_bnd_pt = right_bnd_pt,left_bnd_pt
          #     left_bnd_tangent,right_bnd_tangent = right_bnd_tangent,left_bnd_tangent

          #get affected node data
          ids=samp_to_node_mapping[id]
          for val in ids:
               node_data=node_mapping[val]

               if len(node_data.pt) < len(adj_nodes_mapping[val]):
                    node_data.id.append(id)
                    node_data.pt.append(pt)
                    node_data.radius.append(radius)
                    node_data.theta.append(theta)
                    node_data.phi.append(phi)
                    node_data.speed.append(speed)
                    node_data.plus_pt.append(left_bnd_pt)
                    node_data.minus_pt.append(right_bnd_pt)
                    node_data.plus_theta.append(left_bnd_tangent)
                    node_data.minus_theta.append(right_bnd_tangent)

#process edge information 
def read_edge_header(sample_line,lines,sample_data,edge_offset,numb_edges):
     numb_mappings=getLengthSampNode()
     edge_start=sample_line+numb_mappings*sample_data+edge_offset
     numb_samples=set()
     degen_edges=0
     for idx in range(edge_start,edge_start+numb_edges):
          text=lines[idx]
          starts=[match.start() for match in re.finditer(re.escape('['), text)]
          end=[match.start() for match in re.finditer(re.escape(']'), text)]
          results=list(zip(starts,end))
          source,target=text[results[0][0]+1:results[0][1]].split(' ')
          edge_samples=text[results[1][0]+1:results[1][1]].split(' ')
          del edge_samples[-1]
          if len(edge_samples)==1:
               degen_edges+=1

          if (source,target) not in edge_to_samples:
              edge_mapping[int(source)].append(int(target))
              edge_to_samples[(source,target)]=edge_samples
          else:
              print('@:Seen ',(source,target),' flipping')
              edge_mapping[int(target)].append(int(source))
              edge_samples.reverse()
              edge_to_samples[(target,source)]=edge_samples
              revisit_nodes.add(source)
              revisit_nodes.add(target)
              
          numb_samples.update(edge_samples)
          
     return len(numb_samples),degen_edges

#process edge samples
def read_edge_samples(sample_line,lines,sample_data,edge_offset,numb_edges,
                      numb_edge_samples,degen_edges,node_info):
     
     numb_mappings=getLengthSampNode()
     start=sample_line+numb_mappings*sample_data+edge_offset*2+numb_edges
     end=start+(numb_edge_samples-degen_edges
                -((numb_edges-degen_edges)*2))*sample_data

     for ln in range(start,end,sample_data):
          
          idx=list(range(ln+1,ln+1+node_info))
          
          # get sample id
          text=lines[idx[0]]
          id=int(text.split(' ')[1])

          # get location and radius
          text=lines[idx[1]]
          starts=[match.start() for match in re.finditer(re.escape('('), text)]
          end=[match.start() for match in re.finditer(re.escape(')'), text)]
          results=list(zip(starts,end))
          x,y,radius=text[results[1][0]+1:results[1][1]].split(',')
          pt=(float(x),float(y))
          radius=float(radius)
               
          # get theta
          text=lines[idx[5]]
          theta=np.deg2rad(float(text.split(' ')[1]))

          # get speed , convert to phi
          text=lines[idx[6]]
          speed=float(text.split(' ')[1])
          phi=speedToPhi(speed)

          #get reconstructed boundary points
          left_bnd_pt  = translatePoint(pt,theta+phi, radius)
          right_bnd_pt = translatePoint(pt,theta-phi, radius)

          #get reconstructed boundary tangents
          left_bnd_tangent  = fixAngleMPiPi_new(theta+phi-pi/2.0)
          right_bnd_tangent = fixAngleMPiPi_new(theta-phi+pi/2.0)

          samp=Sample(
               id=id,
               pt=pt,
               type=0.0,
               radius=radius,
               theta=theta,speed=speed,phi=phi,
               plus_pt=left_bnd_pt,minus_pt=right_bnd_pt,
               plus_theta=left_bnd_tangent, minus_theta=right_bnd_tangent)

          edge_samples[id]=samp

          
def compute_sorted_order():

     # define a sorting order for nodes in graph
     key_tuples=[]
     for keys in node_mapping:
          key_tuples.append((node_mapping[keys].pt[0][0],
                             node_mapping[keys].pt[0][1],
                             node_mapping[keys].radius[0],
                             keys))

     sorted_tuples=sorted(key_tuples,key=itemgetter(1,0,2),reverse=False)

     for idx in range(0,len(sorted_tuples)):
          key=sorted_tuples[idx][3]
          value=idx
          node_order[key]=value
          order_mapping[value]=key
          
def compute_adj_feature_matrix(edge_features,NI,NJ,color_space):

     # numb nodes
     numb_nodes=len(node_mapping)
     
     #create adjacency matrix
     adj_matrix=np.zeros((numb_nodes,numb_nodes))

     #create feature matrix
     feature_matrix=np.zeros((numb_nodes,edge_features))

     #create edge matrix
     edge_matrices=defaultdict(list)

     #node locations
     locations =[]
     highest_degree_nodes=[]
     
     #populate adj_matrix
     for key in edge_mapping:
          adj_edges=edge_mapping[key]
          source_idx=node_order[key]
          for item in adj_edges:
               target_idx=node_order[item]
               adj_matrix[source_idx][target_idx]=1

     #write out feature matrix
     for key in node_mapping:
          row=node_order[key]
          
          order_list=[]
          for value in adj_nodes_mapping[key]:
               order_list.append(node_mapping[value].radius[0])
          rad_list=np.argsort(order_list)

          # populate points of node location
          item=node_mapping[key].pt[0]
          feature_matrix[row,0]=item[1]
          feature_matrix[row,1]=item[0]
              
          locations.append(item)
          degree=len(node_mapping[key].theta)
          if  degree >= highest_degree:
               highest_degree_nodes.append(item)
          
          #populate radius of node
          item=node_mapping[key].radius[0]
          feature_matrix[row,2]=item

          #populate theta of node
          start=3
          for idx in range(0,min(len(node_mapping[key].theta),3)):
               item=node_mapping[key].theta[rad_list[idx]]
               feature_matrix[row,idx+start]=item/(2.0*pi)

          #populate phi of node
          start=6
          for idx in range(0,min(len(node_mapping[key].phi),3)):
               item=node_mapping[key].phi[rad_list[idx]]
               feature_matrix[row,idx+start]=item/pi

          # populate left_boundary_points of node
          start=9
          for idx in range(0,min(len(node_mapping[key].plus_pt),3)):
               item=node_mapping[key].plus_pt[rad_list[idx]]
               feature_matrix[row,0+idx*2+start]=item[1]
               feature_matrix[row,1+idx*2+start]=item[0]

          # populate right_boundary_points of node
          start=15
          for idx in range(0,min(len(node_mapping[key].minus_pt),3)):
               item=node_mapping[key].minus_pt[rad_list[idx]]
               feature_matrix[row,0+idx*2+start]=item[1]
               feature_matrix[row,1+idx*2+start]=item[0]

          #populate plus_theta of node
          start=21
          for idx in range(0,min(len(node_mapping[key].plus_theta),3)):
               item=node_mapping[key].plus_theta[rad_list[idx]]
               feature_matrix[row,idx+start]=item/pi

          #populate minus_theta of node
          start=24
          for idx in range(0,min(len(node_mapping[key].minus_theta),3)):
               item=node_mapping[key].minus_theta[rad_list[idx]]
               feature_matrix[row,idx+start]=item/pi

          # populate data type
          start=27
          label=node_mapping[key].type
          if label == 'A':
               out=0.0
          elif label == 'S':
               out=0.25
          elif label == 'F':
               out=0.50
          elif label == 'J':
               out=0.75
          else:
               out=1.0
          feature_matrix[row,start]=out

          start=28
          for idx in range(0,min(len(rad_list),3)):
               source=str(key)
               target=str(adj_nodes_mapping[key][rad_list[idx]])
               pair=(source,target)
               if pair not in curve_stats.keys():
                    pair=(target,source)                        
               props=curve_stats[pair]
               source_idx=node_order[int(pair[0])]
               target_idx=node_order[int(pair[1])]

               ef=[props.SCurve,props.SLength,props.SAngle,props.PCurve,props.PLength,props.PAngle,props.MCurve,props.MLength,props.MAngle,props.PolyArea,props.AvgColor]

               edge_matrices[(source_idx,target_idx)]=ef
               
               feature_matrix[row,0+start]=props.SCurve
               feature_matrix[row,1+start]=props.SLength
               feature_matrix[row,2+start]=props.SAngle
               feature_matrix[row,3+start]=props.PCurve
               feature_matrix[row,4+start]=props.PLength
               feature_matrix[row,5+start]=props.PAngle
               feature_matrix[row,6+start]=props.MCurve
               feature_matrix[row,7+start]=props.MLength
               feature_matrix[row,8+start]=props.MAngle
               feature_matrix[row,9+start]=props.PolyArea
               feature_matrix[row,10+start]=props.AvgColor[0]
               feature_matrix[row,11+start]=props.AvgColor[1]
               feature_matrix[row,12+start]=props.AvgColor[2]

               start=start+13

     return adj_matrix,feature_matrix,edge_matrices


def compute_adj_feature_matrix_hist(edge_features,NI,NJ,color_space,numb_bins):

     # numb nodes
     numb_nodes=len(node_mapping)
     
     #create adjacency matrix
     adj_matrix=np.zeros((numb_nodes,numb_nodes))

     #create feature matrix
     feature_matrix=np.zeros((numb_nodes,edge_features))

     #create edge matrix
     edge_matrices=defaultdict(list)

     #populate adj_matrix
     for key in edge_mapping:
          adj_edges=edge_mapping[key]
          source_idx=node_order[key]
          for item in adj_edges:
               target_idx=node_order[item]
               adj_matrix[source_idx][target_idx]=1

     #write out feature matrix
     for key in node_mapping:
          row=node_order[key]
          
          order_list=[]
          for value in adj_nodes_mapping[key]:
               order_list.append(node_mapping[value].radius[0])
          rad_list=np.argsort(order_list)

          # populate points of node location
          item=node_mapping[key].pt[0]
          feature_matrix[row,0]=item[1]
          feature_matrix[row,1]=item[0]
          
          #populate radius of node
          item=node_mapping[key].radius[0]
          feature_matrix[row,2]=item
          
          # populate data type
          item=node_mapping[key].type
          if item == 'A':
               out=0.0
          elif item == 'S':
               out=0.25
          elif item == 'F':
               out=0.50
          elif item == 'J':
               out=0.75
          else:
               out=1.0
          feature_matrix[row,3]=out

          start=4
          # populate histograms of node
          if node_mapping[key].radius[0]:
              samples=sample_node(node_mapping[key].pt[0],node_mapping[key].radius[0])
              node_hist=get_histogram(samples,color_space,numb_bins)

              feature_matrix[row,(start+numb_bins*0):(start+numb_bins*1)]=node_hist[0]
              feature_matrix[row,(start+numb_bins*1):(start+numb_bins*2)]=node_hist[1]
              feature_matrix[row,(start+numb_bins*2):(start+numb_bins*3)]=node_hist[2]

              gg=np.vstack((feature_matrix[row,(start+numb_bins*0):(start+numb_bins*1)],
                            feature_matrix[row,(start+numb_bins*1):(start+numb_bins*2)],
                            feature_matrix[row,(start+numb_bins*2):(start+numb_bins*3)]))

              # np.savetxt(str(key)+'_samples.txt',samples)
              # np.savetxt(str(key)+'_hist.txt',gg)

          start+=numb_bins*3
          for idx in range(0,min(len(rad_list),3)):
               source=str(key)
               target=str(adj_nodes_mapping[key][rad_list[idx]])
               pair=(source,target)
               if pair not in curve_stats.keys():
                    pair=(target,source)                        
               props=curve_stats[pair]
               source_idx=node_order[int(pair[0])]
               target_idx=node_order[int(pair[1])]

               ef=[props.SCurve,props.SLength,props.SAngle,props.PCurve,props.PLength,props.PAngle,props.MCurve,props.MLength,props.MAngle,props.PolyArea,props.HistColor]

               edge_matrices[(source_idx,target_idx)]=ef
               
               feature_matrix[row,0+start]=props.SCurve
               feature_matrix[row,1+start]=props.SLength
               feature_matrix[row,2+start]=props.SAngle
               feature_matrix[row,3+start]=props.PCurve
               feature_matrix[row,4+start]=props.PLength
               feature_matrix[row,5+start]=props.PAngle
               feature_matrix[row,6+start]=props.MCurve
               feature_matrix[row,7+start]=props.MLength
               feature_matrix[row,8+start]=props.MAngle
               feature_matrix[row,9+start]=props.PolyArea
               start+=10
               
               feature_matrix[row,(start+numb_bins*0):(start+numb_bins*1)]=props.HistColor[0]
               feature_matrix[row,(start+numb_bins*1):(start+numb_bins*2)]=props.HistColor[1]
               feature_matrix[row,(start+numb_bins*2):(start+numb_bins*3)]=props.HistColor[2]

               # gg=np.vstack((feature_matrix[row,(start+numb_bins*0):(start+numb_bins*1)],
               #               feature_matrix[row,(start+numb_bins*1):(start+numb_bins*2)],
               #               feature_matrix[row,(start+numb_bins*2):(start+numb_bins*3)]))
                            
               # np.savetxt(pair[0]+'_s_to_t_'+pair[1]+'_hist.txt',gg)
               start+=numb_bins*3

     return adj_matrix,feature_matrix,edge_matrices
    
def convertEsfFile(esf_file,image_file,con_file,coarse,numb_bins):

     I=cv2.imread(image_file)
     NI=I.shape[0]
     NJ=I.shape[1]
     dims=np.array([NI,NJ])

     con_points=read_cemv_file(con_file)
          
     file=open(esf_file,'r')
     lines= file.read().splitlines() 

     node_line=15
     sample_data=12
     node_info=9
     edge_offset=4
     
     _,numb_nodes=lines[7].split(':')
     _,numb_edges=lines[8].split(':')

     numb_nodes=int(numb_nodes)
     numb_edges=int(numb_edges)

     sample_line=node_line+numb_nodes+4

     read_node_header(node_line,lines,numb_nodes)
     read_node_samples(sample_line,lines,sample_data,node_info)
     numb_edge_samples,numb_degen_edges=read_edge_header(sample_line,lines,
                                                         sample_data,
                                                         edge_offset,numb_edges)
     read_edge_samples(sample_line,lines,sample_data,edge_offset,numb_edges,
                       numb_edge_samples,numb_degen_edges,node_info)
     compute_edge_stats(I,numb_bins)
     compute_sorted_order()


     if numb_bins > 0 :
         print('Computing hist features along geometric features')
         edge_features=4+3*numb_bins+(10+3*numb_bins)*3
         adj_matrix,feature_matrix,edge_matrices=compute_adj_feature_matrix_hist(
             edge_features,NI,NJ,I,numb_bins)
     else:
         print('Computing Original Stats')
         edge_features=67
         adj_matrix,feature_matrix,edge_matrices=compute_adj_feature_matrix(
             edge_features,NI,NJ,I)

         # get shape context
         a = SC()
         sg_points=np.copy(feature_matrix[:,:2])
         sg_points[np.abs(sg_points)<1.0e-10]=0.0
         query=list(map(tuple,sg_points))
         P=a.compute(query,con_points)

         # get colors
         colors=get_pixel_values(feature_matrix,I)
     
         feature_matrix=np.concatenate((feature_matrix,P,colors),axis=1)

     
     if coarse:
         print('Coarsening Graph')
         G,paths=get_paths(adj_matrix,I)
         flag=check_paths(G,paths)
         if flag:
             print('All paths check out!! good!')
         else:
             print('ERRORPATH: not all paths checked')
         coarsen_graph(paths)
         adj_matrix,feature_matrix,edge_matrices=compute_adj_feature_matrix(edge_features,NI,NJ,I)
         #vis(G,I,feature_matrix[:,:2],paths)

         
     nodes=adj_matrix.shape[0]
     
     fname_graph=os.path.splitext(esf_file)[0]+'-n'+str(nodes)+'-shock_graph.h5'
     hf = h5py.File(fname_graph,'w')

     hf.create_dataset('feature',data=feature_matrix)
     hf.create_dataset('adj_matrix',data=adj_matrix)
     hf.create_dataset('dims',data=dims)


     if numb_bins > 0:
         print('Saving edge features with hist')
         output_edge=np.zeros((len(edge_matrices),2+10+3*numb_bins))
         idx=0
         for key,value in edge_matrices.items():
             output_edge[idx,0]=key[0]
             output_edge[idx,1]=key[1]
             output_edge[idx,2]=value[0]
             output_edge[idx,3]=value[1]
             output_edge[idx,4]=value[2]
             output_edge[idx,5]=value[3]
             output_edge[idx,6]=value[4]
             output_edge[idx,7]=value[5]
             output_edge[idx,8]=value[6]
             output_edge[idx,9]=value[7]
             output_edge[idx,10]=value[8]
             output_edge[idx,11]=value[9]

             start=12
             output_edge[idx,(start+numb_bins*0):(start+numb_bins*1)]=value[10][0]
             output_edge[idx,(start+numb_bins*1):(start+numb_bins*2)]=value[10][1]
             output_edge[idx,(start+numb_bins*2):(start+numb_bins*3)]=value[10][2]
             idx=idx+1
     else:
         print('Saving edge features with avg color')
         output_edge=np.zeros((len(edge_matrices),15))
         idx=0
         for key,value in edge_matrices.items():
             output_edge[idx,0]=key[0]
             output_edge[idx,1]=key[1]
             output_edge[idx,2]=value[0]
             output_edge[idx,3]=value[1]
             output_edge[idx,4]=value[2]
             output_edge[idx,5]=value[3]
             output_edge[idx,6]=value[4]
             output_edge[idx,7]=value[5]
             output_edge[idx,8]=value[6]
             output_edge[idx,9]=value[7]
             output_edge[idx,10]=value[8]
             output_edge[idx,11]=value[9]
             output_edge[idx,12]=value[10][0]
             output_edge[idx,13]=value[10][1]
             output_edge[idx,14]=value[10][2]
             idx=idx+1          
         
     hf.create_dataset('edge_features',data=output_edge)
         
     hf.close()
     
if __name__ == '__main__':
     parser = argparse.ArgumentParser(description='Convert SG File')

     parser.add_argument("--esf",type=str,help='Shock Graph file')
     parser.add_argument("--image",type=str,help='Image File')
     parser.add_argument("--con",type=str,help='Contour File')
     parser.add_argument("--bins",type=int,default=11,help='Number of bins for histograms')
     parser.add_argument("--coarse",type=bool,default=False,
                         help="Coarse graph with only degree 1 and 3 nodes")
     args=parser.parse_args()
     
     convertEsfFile(args.esf,args.image,args.con,args.coarse,args.bins)
     
