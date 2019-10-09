import re
import sys
import os
import scipy.misc
import numpy as np
import h5py
import math

from collections import defaultdict
from collections import namedtuple
from math import acos
from math import pi
from math import cos
from math import sin
from operator import itemgetter
from scipy.spatial.distance import cdist

node_order=dict()
node_mapping=dict()
edge_samples=dict()
edge_to_samples=dict()
curve_stats=dict()
samp_to_node_mapping=defaultdict(list)
edge_mapping=defaultdict(list)
adj_nodes_mapping=defaultdict(list)
Sample=namedtuple('Sample','pt type radius theta phi plus_pt \
minus_pt plus_theta minus_theta')
CurveProps=namedtuple('CurveProps','SCurve SLength SAngle PCurve PLength PAngle MCurve MLength MAngle PolyArea')
highest_degree=0
ZERO_TOLERANCE=1E-1


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
     angle.append(0.0);
     totalAngleChange=0.0;

     px=curve[0][0]
     py=curve[0][1]
     for i in range(1,len(curve)):
          cx=curve[i][0]
          cy=curve[i][1]
          theta=math.atan2(cy-py,cx-px);
          angle.append(theta);
          px=cx
          py=cy

     if len(curve) >2 :
          angle[0]=angle[1];
          for i in range(1,len(angle)):
               totalAngleChange += abs(angle[i]-angle[i-1]);

     return totalAngleChange
  
def computeCurveStats(curve):

     if len(curve)==0:
          return 0.0,0.0,0.0
     
     arcLength,length=computeArcLength(curve)
     dx,dy=computeDerivatives(curve)
     _,totalCurvature=computeCurvatures(dx,dy,arcLength,len(curve))
     totalAngleChange=computeAngles(curve)

     return length,totalCurvature,totalAngleChange


def compute_edge_stats():

    
    for key,value in edge_to_samples.items():
          
          first_pt=node_mapping[int(key[0])].pt[0]
          shock_curve=[edge_samples.get(int(value[id])).pt for id in range(1,len(value)-1)]
          last_pt=node_mapping[int(key[1])].pt[0]
          shock_curve.insert(0,first_pt)
          shock_curve.append(last_pt)

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
               totals=zip(*poly)
               area=polyArea(np.array(totals[0]),np.array(totals[1]))

          stats=CurveProps(SCurve=shock_totalCurvature,
                           SLength=shock_length,
                           SAngle=shock_angle,
                           PCurve=plus_totalCurvature,
                           PLength=plus_length,
                           PAngle=plus_angle,
                           MCurve=minus_totalCurvature,
                           MLength=minus_length,
                           MAngle=minus_angle,
                           PolyArea=area)
          
          curve_stats[key]=stats

          
def getLengthSampNode():
     length=0
     for value in samp_to_node_mapping.itervalues():
          length=length+len(value)
     return length
     
def speedToPhi(speed):
     if speed != 0 and speed < 99990:
          phi = acos(-1.0/speed)
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
          samp=Sample(pt=[],
                      type=label,
                      radius=[],
                      theta=[],phi=[],
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


          #get affected node data
          ids=samp_to_node_mapping[id]
          for val in ids:
               node_data=node_mapping[val]

               if len(node_data.pt) < len(adj_nodes_mapping[val]):
                    node_data.pt.append(pt)
                    node_data.radius.append(radius)
                    node_data.theta.append(theta)
                    node_data.phi.append(phi)
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
          edge_mapping[int(source)].append(int(target))
          edge_to_samples[(source,target)]=edge_samples
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
               pt=pt,
               type=0.0,
               radius=radius,
               theta=theta,phi=phi,
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

     sorted_tuples=sorted(key_tuples,key=itemgetter(1,0,3),reverse=False)

     for idx in range(0,len(sorted_tuples)):
          key=sorted_tuples[idx][3]
          value=idx
          node_order[key]=value

def compute_adj_feature_matrix(edge_features,NI,NJ):

     # numb nodes
     numb_nodes=len(node_mapping)
     
     #create adjacency matrix
     adj_matrix=np.zeros((numb_nodes,numb_nodes))

     #create feature matrix
     feature_matrix=np.zeros((numb_nodes,edge_features))

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
          feature_matrix[row][0]=item[1]
          feature_matrix[row][1]=item[0]
              
          locations.append(item)
          degree=len(node_mapping[key].theta)
          if  degree >= highest_degree:
               highest_degree_nodes.append(item)
          
          #populate radius of node
          item=node_mapping[key].radius[0]
          feature_matrix[row][2]=item

          #populate theta of node
          start=3
          for idx in range(0,min(len(node_mapping[key].theta),3)):
               item=node_mapping[key].theta[rad_list[idx]]
               feature_matrix[row][idx+start]=item/(2.0*pi)

          #populate phi of node
          start=6
          for idx in range(0,min(len(node_mapping[key].phi),3)):
               item=node_mapping[key].phi[rad_list[idx]]
               feature_matrix[row][idx+start]=item/pi

          # populate left_boundary_points of node
          start=9
          for idx in range(0,min(len(node_mapping[key].plus_pt),3)):
               item=node_mapping[key].plus_pt[rad_list[idx]]
               feature_matrix[row][0+idx*2+start]=item[1]
               feature_matrix[row][1+idx*2+start]=item[0]

          # populate right_boundary_points of node
          start=15
          for idx in range(0,min(len(node_mapping[key].minus_pt),3)):
               item=node_mapping[key].minus_pt[rad_list[idx]]
               feature_matrix[row][0+idx*2+start]=item[1]
               feature_matrix[row][1+idx*2+start]=item[0]
          
          
          #populate plus_theta of node
          start=21
          for idx in range(0,min(len(node_mapping[key].plus_theta),3)):
               item=node_mapping[key].plus_theta[rad_list[idx]]
               feature_matrix[row][idx+start]=item/pi

          #populate minus_theta of node
          start=24
          for idx in range(0,min(len(node_mapping[key].minus_theta),3)):
               item=node_mapping[key].minus_theta[rad_list[idx]]
               feature_matrix[row][idx+start]=item/pi

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
          feature_matrix[row][start]=out

          start=28
          for idx in range(0,min(len(rad_list),3)):
               source=str(key)
               target=str(adj_nodes_mapping[key][rad_list[idx]])
               pair=(source,target)
               if pair not in curve_stats.keys():
                    pair=(target,source)                        
               props=curve_stats[pair]
                   
               feature_matrix[row][0+start]=props.SCurve
               feature_matrix[row][1+start]=props.SLength
               feature_matrix[row][2+start]=props.SAngle
               feature_matrix[row][3+start]=props.PCurve
               feature_matrix[row][4+start]=props.PLength
               feature_matrix[row][5+start]=props.PAngle
               feature_matrix[row][6+start]=props.MCurve
               feature_matrix[row][7+start]=props.MLength
               feature_matrix[row][8+start]=props.MAngle
               feature_matrix[row][9+start]=props.PolyArea
               start=start+10

     sorted_locations=sorted(locations,key=itemgetter(0,1),reverse=False)
     ul_corner=sorted_locations[0]
     lr_corner=sorted_locations[-1]
     center=((ul_corner[0]+lr_corner[0])/2.0,(ul_corner[1]+lr_corner[1])/2.0)
     high_order_nodes=np.array(highest_degree_nodes)
     temp=np.zeros((1,2))
     temp[0][0]=center[0]
     temp[0][1]=center[1]
     distances=np.squeeze(cdist(high_order_nodes,temp))
     reference_ind=np.argsort(distances)
     reference_pt=high_order_nodes[reference_ind[0]]

     feature_matrix[:,:2]-=reference_pt

     zero_set=np.array([0.0,0.0])
     
     for row_idx in range(0,feature_matrix.shape[0]):
          feature_matrix[row_idx,9:11]-=reference_pt
          feature_matrix[row_idx,15:17]-=reference_pt

          if np.array_equal(feature_matrix[row_idx,11:13],zero_set)==False:
               feature_matrix[row_idx,11:13]-=reference_pt
               feature_matrix[row_idx,17:19]-=reference_pt

          if np.array_equal(feature_matrix[row_idx,13:15],zero_set)==False:
               feature_matrix[row_idx,13:15]-=reference_pt
               feature_matrix[row_idx,19:21]-=reference_pt

     max_offsets=np.amax(np.abs(feature_matrix[:,:2]),axis=0)
     max_radius=np.amax(feature_matrix,axis=0)[2]

     feature_matrix[:,:2] /= max_offsets
     feature_matrix[:,2] /= max_radius
     feature_matrix[:,9:11] /= max_offsets
     feature_matrix[:,15:17] /= max_offsets
     feature_matrix[:,11:13] /= max_offsets
     feature_matrix[:,17:19] /= max_offsets
     feature_matrix[:,13:15] /= max_offsets
     feature_matrix[:,19:21] /= max_offsets

     debug=np.concatenate((reference_pt,max_offsets,np.array([max_radius])),
                          axis=0)
     return adj_matrix,feature_matrix,debug
    
def convertEsfFile(esf_file,image_file):

     I=scipy.misc.imread(image_file)
     NI=I.shape[0]
     NJ=I.shape[1]
     dims=np.array([NI,NJ])
     
     file=open(esf_file,'r')
     lines= file.read().splitlines() 

     node_line=15
     sample_data=12
     node_info=9
     edge_offset=4
     edge_features=58
     
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
     compute_edge_stats()
     compute_sorted_order()
     adj_matrix,feature_matrix,ref_point=\
     compute_adj_feature_matrix(edge_features,NI,NJ)

     nodes=adj_matrix.shape[0]
     
     fname_graph=os.path.splitext(esf_file)[0]+'-n'+str(nodes)+'-shock_graph.h5'
     hf = h5py.File(fname_graph,'w')

     hf.create_dataset('feature',data=feature_matrix)
     hf.create_dataset('adj_matrix',data=adj_matrix)
     hf.create_dataset('debug',data=ref_point)
     hf.create_dataset('dims',data=dims)
     hf.close()

     
if __name__ == '__main__':
     esf_file=sys.argv[1]
     image_file=sys.argv[2]
     convertEsfFile(esf_file,image_file)
