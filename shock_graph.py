import re
import sys
import os
import scipy.misc
import numpy as np

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
samp_to_node_mapping=dict()
edge_mapping=defaultdict(list)
Sample=namedtuple('Sample','pt type radius theta phi plus_pt \
minus_pt plus_theta minus_theta')
     
def speedToPhi(speed):
     if speed != 0 and speed < 99990:
          phi = acos(-1.0/speed);
     else:
          phi = pi/2.0
     return phi

def fixAngleMPiPi_new(a):
     if a < -pi:
          return a+2.0*pi
     elif a > pi:
          return a-2.0*pi;
     else:
          return a

def translatePoint(pt,v,length):
     x = pt[0] + length*cos(v);
     y = pt[1] + length*sin(v);
     return (x,y)


# process node header data
def read_node_header(node_line,lines,numb_nodes):

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
          results=zip(starts,ends)
          adj_nodes=text[results[0][0]+1:results[0][1]].split(' ')
          adj_samples=text[results[1][0]+1:results[1][1]].split(' ')
          for key in adj_samples:
               if len(key):
                    samp_to_node_mapping[int(key)]=int(node_id)

# process node sample data
def read_node_samples(sample_line,lines,sample_data,node_info):

     numb_mappings=len(samp_to_node_mapping)
     start=sample_line
     end=sample_line+numb_mappings*sample_data
     for ln in range(start,end,sample_data):
          
          idx=range(ln+1,ln+1+node_info)
          
          # get sample id
          text=lines[idx[0]]
          id=int(text.split(' ')[1])

          # get location and radius
          text=lines[idx[1]]
          starts=[match.start() for match in re.finditer(re.escape('('), text)]
          end=[match.start() for match in re.finditer(re.escape(')'), text)]
          results=zip(starts,end)
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
          left_bnd_pt  = translatePoint(pt,theta+phi, radius);
          right_bnd_pt = translatePoint(pt,theta-phi, radius);

          #get reconstructed boundary tangents
          left_bnd_tangent  = fixAngleMPiPi_new(theta+phi-pi/2.0);
          right_bnd_tangent = fixAngleMPiPi_new(theta-phi+pi/2.0);

          #get affected node data
          node_data=node_mapping[samp_to_node_mapping[id]]

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
     numb_mappings=len(samp_to_node_mapping)
     edge_start=sample_line+numb_mappings*sample_data+edge_offset
     for idx in range(edge_start,edge_start+numb_edges):
          text=lines[idx]
          starts=[match.start() for match in re.finditer(re.escape('['), text)]
          end=[match.start() for match in re.finditer(re.escape(']'), text)]
          results=zip(starts,end)
          source,target=text[results[0][0]+1:results[0][1]].split(' ')
          edge_mapping[int(source)].append(int(target))


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
     degree_three_nodes=[]
     
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

          # populate points of node location
          item=node_mapping[key].pt[0]
          feature_matrix[row][0]=item[1]
          feature_matrix[row][1]=item[0]

          locations.append(item)
          degree=len(node_mapping[key].theta)
          if  degree >= 3:
               degree_three_nodes.append(item)
          
          #populate radius of node
          item=node_mapping[key].radius[0]
          feature_matrix[row][2]=item

          #populate theta of node
          start=3
          for idx in range(0,len(node_mapping[key].theta)):
               item=node_mapping[key].theta[idx]
               feature_matrix[row][idx+start]=item/(2.0*pi)

          #populate phi of node
          start=6
          for idx in range(0,len(node_mapping[key].phi)):
               item=node_mapping[key].phi[idx]
               feature_matrix[row][idx+start]=item/pi

          # populate left_boundary_points of node
          start=9
          for idx in range(0,len(node_mapping[key].plus_pt)):
               item=node_mapping[key].plus_pt[idx]
               feature_matrix[row][0+idx*2+start]=item[1]
               feature_matrix[row][1+idx*2+start]=item[0]
          
          #populate plus_theta of node
          start=15
          for idx in range(0,len(node_mapping[key].plus_theta)):
               item=node_mapping[key].plus_theta[idx]
               feature_matrix[row][idx+start]=item/pi

          # populate data type
          start=18
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


     sorted_locations=sorted(locations,key=itemgetter(0,1),reverse=False)
     ul_corner=sorted_locations[0]
     lr_corner=sorted_locations[-1]
     center=((ul_corner[0]+lr_corner[0])/2.0,(ul_corner[1]+lr_corner[1])/2.0)
     high_order_nodes=np.array(degree_three_nodes)
     temp=np.zeros((1,2))
     temp[0][0]=center[0]
     temp[0][1]=center[1]
     distances=np.squeeze(cdist(high_order_nodes,temp))
     reference_ind=np.argsort(distances)
     reference_pt=high_order_nodes[reference_ind[0]]

     feature_matrix[:,:2]-=reference_pt
     feature_matrix[:,9:11]-=reference_pt
     feature_matrix[:,11:13]-=reference_pt
     feature_matrix[:,13:15]-=reference_pt

     max_offsets=np.amax(np.abs(feature_matrix[:,:2]),axis=0)
     max_radius=np.amax(feature_matrix,axis=0)[2]

     feature_matrix[:,:2] /= max_offsets
     feature_matrix[:,2] /= max_radius
     feature_matrix[:,9:11] /= max_offsets
     feature_matrix[:,11:13] /= max_offsets
     feature_matrix[:,13:15] /= max_offsets

     debug=np.concatenate((reference_pt,max_offsets,np.array([max_radius])),
                          axis=0)
     return adj_matrix,feature_matrix,debug
    
def convertEsfFile(esf_file,image_file):

     I=scipy.misc.imread(image_file)
     NI=I.shape[0]
     NJ=I.shape[1]
     
     file=open(esf_file,'r')
     lines= file.read().splitlines() 

     node_line=15
     sample_data=12
     node_info=9
     edge_offset=4
     edge_features=19
     
     _,numb_nodes=lines[7].split(':')
     _,numb_edges=lines[8].split(':')

     numb_nodes=int(numb_nodes)
     numb_edges=int(numb_edges)

     sample_line=node_line+numb_nodes+4

     read_node_header(node_line,lines,numb_nodes)
     read_node_samples(sample_line,lines,sample_data,node_info)
     read_edge_header(sample_line,lines,sample_data,edge_offset,numb_edges)
     compute_sorted_order()
     adj_matrix,feature_matrix,ref_point=\
     compute_adj_feature_matrix(edge_features,NI,NJ)

     fname_feature=os.path.splitext(esf_file)[0]+'_feature.txt'
     fname_adj_matrix=os.path.splitext(esf_file)[0]+'_adj_matrix.txt'
     fname_debug_info=os.path.splitext(esf_file)[0]+'_debug.txt'
     
     np.savetxt(fname_feature,feature_matrix,delimiter=' ')
     np.savetxt(fname_adj_matrix,adj_matrix,delimiter=' ')
     np.savetxt(fname_debug_info,ref_point,delimiter=' ')
     
if __name__ == '__main__':
     esf_file=sys.argv[1]
     image_file=sys.argv[2]
     convertEsfFile(esf_file,image_file)
