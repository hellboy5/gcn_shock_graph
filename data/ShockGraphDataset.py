import numpy as np
import glob
import random
import h5py
import os
import re
import dgl
import torch
import scipy.sparse as sp

import math

from tqdm import tqdm
from torch.utils.data.dataset import Dataset
#from scipy.misc import imread
from scipy import ndimage
from operator import itemgetter
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.shape_context import SC

fmnist_map={'top':0,'trouser':1,'pullover':2,'dress':3,'coat':4,'sandal':5,'shirt':6,'sneaker':7,'bag':8,'ankleboot':9}

stl10_map={'airplane':0, 'bird':1,'car':2,'cat':3,'deer':4,'dog':5,'horse':6,'monkey':7,'ship':8,'truck':9}

cifar100_map={'couch': 25, 'pine_tree': 59, 'butterfly': 14, 'mountain': 49, 'bus': 13, 'sunflower': 82, 'beetle': 7, 'lamp': 40, 'can': 16, 'beaver': 4, 'bee': 6, 'elephant': 31, 'kangaroo': 38, 'oak_tree': 52, 'orchid': 54, 'trout': 91, 'pickup_truck': 58, 'chimpanzee': 21, 'worm': 99, 'snail': 77, 'television': 87, 'hamster': 36, 'squirrel': 80, 'lion': 43, 'otter': 55, 'shrew': 74, 'pear': 57, 'turtle': 93, 'seal': 72, 'plate': 61, 'fox': 34, 'train': 90, 'porcupine': 63, 'tulip': 92, 'keyboard': 39, 'orange': 53, 'possum': 64, 'cattle': 19, 'skyscraper': 76, 'bear': 3, 'cup': 28, 'cockroach': 24, 'caterpillar': 18, 'mouse': 50, 'forest': 33, 'rabbit': 65, 'aquarium_fish': 1, 'chair': 20, 'castle': 17, 'palm_tree': 56, 'telephone': 86, 'mushroom': 51, 'streetcar': 81, 'willow_tree': 96, 'man': 46, 'wardrobe': 94, 'bowl': 10, 'sweet_pepper': 83, 'maple_tree': 47, 'snake': 78, 'whale': 95, 'poppy': 62, 'tank': 85, 'bed': 5, 'rose': 70, 'crocodile': 27, 'raccoon': 66, 'tractor': 89, 'bicycle': 8, 'bridge': 12, 'dinosaur': 29, 'crab': 26, 'clock': 22, 'bottle': 9, 'lawn_mower': 41, 'road': 68, 'spider': 79, 'skunk': 75, 'tiger': 88, 'sea': 71, 'lizard': 44, 'cloud': 23, 'ray': 67, 'house': 37, 'lobster': 45, 'boy': 11, 'plain': 60, 'table': 84, 'dolphin': 30, 'camel': 15, 'rocket': 69, 'baby': 2, 'girl': 35, 'shark': 73, 'motorcycle': 48, 'flatfish': 32, 'leopard': 42, 'wolf': 97, 'apple': 0, 'woman': 98}

imageclef_map={'aeroplane':0, 'bike':1, 'bird':2, 'boat':3, 'bottle':4, 'bus':5, 'car':6, 'dog':7, 'horse':8, 'monitor':9, 'motorbike':10, 'people':11}

office31_map={'back_pack':0,
              'bike':1,
              'bike_helmet':2,
              'bookcase':3,
              'bottle':4,
              'calculator':5,
              'desk_chair':6,
              'desk_lamp':7,
              'desktop_computer':8,
              'file_cabinet':9,
              'headphones':10,
              'keyboard':11,
              'laptop_computer':12,
              'letter_tray':13,
              'mobile_phone':14,
              'monitor':15,
              'mouse':16,
              'mug':17,
              'paper_notebook':18,
              'pen':19,
              'phone':20,
              'printer':21,
              'projector':22,
              'punchers':23,
              'ring_binder':24,
              'ruler':25,
              'scissors':26,
              'speaker':27,
              'stapler':28,
              'tape_dispenser':29,
              'trash_can':30}


officehome_map={'Alarm_Clock':0,
                'Backpack':1,
                'Batteries':2,
                'Bed':3,
                'Bike':4,
                'Bottle':5,
                'Bucket':6,
                'Calculator':7,
                'Calendar':8,
                'Candles':9,
                'Chair':10,
                'Clipboards':11,
                'Computer':12,
                'Couch':13,
                'Curtains':14,
                'Desk_Lamp':15,
                'Drill':16,
                'Eraser':17,
                'Exit_Sign':18,
                'Fan':19,
                'File_Cabinet':20,
                'Flipflops':21,
                'Flowers':22,
                'Folder':23,
                'Fork':24,
                'Glasses':25,
                'Hammer':26,
                'Helmet':27,
                'Kettle':28,
                'Keyboard':29,
                'Knives':30,
                'Lamp_Shade':31,
                'Laptop':32,
                'Marker':33,
                'Monitor':34,
                'Mop':35,
                'Mouse':36,
                'Mug':37,
                'Notebook':38,
                'Oven':39,
                'Pan':40,
                'Paper_Clip':41,
                'Pen':42,
                'Pencil':43,
                'Postit_Notes':44,
                'Printer':45,
                'Push_Pin':46,
                'Radio':47,
                'Refrigerator':48,
                'Ruler':49,
                'Scissors':50,
                'Screwdriver':51,
                'Shelf':52,
                'Sink':53,
                'Sneakers':54,
                'Soda':55,
                'Speaker':56,
                'Spoon':57,
                'TV':58,
                'Table':59,
                'Telephone':60,
                'ToothBrush':61,
                'Toys':62,
                'Trash_Can':63,
                'Webcam':64}

tign_map={'n01443537':0,
          'n01629819':1,
          'n01641577':2,
          'n01644900':3,
          'n01698640':4,
          'n01742172':5,
          'n01768244':6,
          'n01770393':7,
          'n01774384':8,
          'n01774750':9,
          'n01784675':10,
          'n01855672':11,
          'n01882714':12,
          'n01910747':13,
          'n01917289':14,
          'n01944390':15,
          'n01945685':16,
          'n01950731':17,
          'n01983481':18,
          'n01984695':19,
          'n02002724':20,
          'n02056570':21,
          'n02058221':22,
          'n02074367':23,
          'n02085620':24,
          'n02094433':25,
          'n02099601':26,
          'n02099712':27,
          'n02106662':28,
          'n02113799':29,
          'n02123045':30,
          'n02123394':31,
          'n02124075':32,
          'n02125311':33,
          'n02129165':34,
          'n02132136':35,
          'n02165456':36,
          'n02190166':37,
          'n02206856':38,
          'n02226429':39,
          'n02231487':40,
          'n02233338':41,
          'n02236044':42,
          'n02268443':43,
          'n02279972':44,
          'n02281406':45,
          'n02321529':46,
          'n02364673':47,
          'n02395406':48,
          'n02403003':49,
          'n02410509':50,
          'n02415577':51,
          'n02423022':52,
          'n02437312':53,
          'n02480495':54,
          'n02481823':55,
          'n02486410':56,
          'n02504458':57,
          'n02509815':58,
          'n02666196':59,
          'n02669723':60,
          'n02699494':61,
          'n02730930':62,
          'n02769748':63,
          'n02788148':64,
          'n02791270':65,
          'n02793495':66,
          'n02795169':67,
          'n02802426':68,
          'n02808440':69,
          'n02814533':70,
          'n02814860':71,
          'n02815834':72,
          'n02823428':73,
          'n02837789':74,
          'n02841315':75,
          'n02843684':76,
          'n02883205':77,
          'n02892201':78,
          'n02906734':79,
          'n02909870':80,
          'n02917067':81,
          'n02927161':82,
          'n02948072':83,
          'n02950826':84,
          'n02963159':85,
          'n02977058':86,
          'n02988304':87,
          'n02999410':88,
          'n03014705':89,
          'n03026506':90,
          'n03042490':91,
          'n03085013':92,
          'n03089624':93,
          'n03100240':94,
          'n03126707':95,
          'n03160309':96,
          'n03179701':97,
          'n03201208':98,
          'n03250847':99,
          'n03255030':100,
          'n03355925':101,
          'n03388043':102,
          'n03393912':103,
          'n03400231':104,
          'n03404251':105,
          'n03424325':106,
          'n03444034':107,
          'n03447447':108,
          'n03544143':109,
          'n03584254':110,
          'n03599486':111,
          'n03617480':112,
          'n03637318':113,
          'n03649909':114,
          'n03662601':115,
          'n03670208':116,
          'n03706229':117,
          'n03733131':118,
          'n03763968':119,
          'n03770439':120,
          'n03796401':121,
          'n03804744':122,
          'n03814639':123,
          'n03837869':124,
          'n03838899':125,
          'n03854065':126,
          'n03891332':127,
          'n03902125':128,
          'n03930313':129,
          'n03937543':130,
          'n03970156':131,
          'n03976657':132,
          'n03977966':133,
          'n03980874':134,
          'n03983396':135,
          'n03992509':136,
          'n04008634':137,
          'n04023962':138,
          'n04067472':139,
          'n04070727':140,
          'n04074963':141,
          'n04099969':142,
          'n04118538':143,
          'n04133789':144,
          'n04146614':145,
          'n04149813':146,
          'n04179913':147,
          'n04251144':148,
          'n04254777':149,
          'n04259630':150,
          'n04265275':151,
          'n04275548':152,
          'n04285008':153,
          'n04311004':154,
          'n04328186':155,
          'n04356056':156,
          'n04366367':157,
          'n04371430':158,
          'n04376876':159,
          'n04398044':160,
          'n04399382':161,
          'n04417672':162,
          'n04456115':163,
          'n04465501':164,
          'n04486054':165,
          'n04487081':166,
          'n04501370':167,
          'n04507155':168,
          'n04532106':169,
          'n04532670':170,
          'n04540053':171,
          'n04560804':172,
          'n04562935':173,
          'n04596742':174,
          'n04597913':175,
          'n06596364':176,
          'n07579787':177,
          'n07583066':178,
          'n07614500':179,
          'n07615774':180,
          'n07695742':181,
          'n07711569':182,
          'n07715103':183,
          'n07720875':184,
          'n07734744':185,
          'n07747607':186,
          'n07749582':187,
          'n07753592':188,
          'n07768694':189,
          'n07871810':190,
          'n07873807':191,
          'n07875152':192,
          'n07920052':193,
          'n09193705':194,
          'n09246464':195,
          'n09256479':196,
          'n09332890':197,
          'n09428293':198,
          'n12267677':199}

mign_map={'n01532829':0,
          'n01558993':1,
          'n01704323':2,
          'n01749939':3,
          'n01770081':4,
          'n01843383':5,
          'n01910747':6,
          'n02074367':7,
          'n02089867':8,
          'n02091831':9,
          'n02101006':10,
          'n02105505':11,
          'n02108089':12,
          'n02108551':13,
          'n02108915':14,
          'n02111277':15,
          'n02113712':16,
          'n02120079':17,
          'n02165456':18,
          'n02457408':19,
          'n02606052':20,
          'n02687172':21,
          'n02747177':22,
          'n02795169':23,
          'n02823428':24,
          'n02966193':25,
          'n03017168':26,
          'n03047690':27,
          'n03062245':28,
          'n03207743':29,
          'n03220513':30,
          'n03337140':31,
          'n03347037':32,
          'n03400231':33,
          'n03476684':34,
          'n03527444':35,
          'n03676483':36,
          'n03838899':37,
          'n03854065':38,
          'n03888605':39,
          'n03908618':40,
          'n03924679':41,
          'n03998194':42,
          'n04067472':43,
          'n04243546':44,
          'n04251144':45,
          'n04258138':46,
          'n04275548':47,
          'n04296562':48,
          'n04389033':49,
          'n04435653':50,
          'n04443257':51,
          'n04509417':52,
          'n04515003':53,
          'n04596742':54,
          'n04604644':55,
          'n04612504':56,
          'n06794110':57,
          'n07584110':58,
          'n07697537':59,
          'n07747607':60,
          'n09246464':61,
          'n13054560':62,
          'n13133613':63}

def fixAngleMPiPi_new_vector(vec):
    output=np.zeros(np.shape(vec))
    it=np.nditer(vec,flags=['f_index'])
    for a in it:
        if a < -math.pi:
            output[it.index]=a+2.0*math.pi
        elif a > math.pi:
            output[it.index]=a-2.0*math.pi;
        else:
            output[it.index]=a

    return output

def fixAngle2PiPi_new_vector(vec):
    output=np.zeros(np.shape(vec))
    it=np.nditer(vec,flags=['f_index'])
    for a in it:
        if a < -2.0*math.pi:
            output[it.index]=a+2.0*math.pi
        elif a > 2.0*math.pi:
            output[it.index]=a-2.0*math.pi;
        else:
            output[it.index]=a

    return output

class ShockGraphDataset(Dataset):
    'Generates data for Keras'
    def __init__(self,directory,dataset,norm_factors,node_app=False,edge_app=False,cache=True,symmetric=False,data_augment=False,flip_pp=False,grid=8):
        'Initialization'
        
        self.directory = directory
        self.node_app=node_app
        self.edge_app=edge_app
        self.cache=cache
        self.symmetric = symmetric
        self.files=[]
        self.class_mapping={}
        self.sg_graphs=[]
        self.sg_labels=[]
        self.adj_matrices=[]
        self.width=1
        self.center=np.zeros(2)
        self.image_size=1
        self.sg_features=[]
        self.trans=()
        self.factor=1
        self.flip_pp=flip_pp
        self.data_augment=data_augment
        self.grid=grid
        self.grid_mapping=[]
        self.norm_factors=norm_factors
        self.dataset=dataset
        
        if dataset=='cifar100':
            print('Using cifar 100 dataset')
            self.class_mapping=cifar100_map
        elif dataset=='stl10':
            print('Using stl 10 dataset')
            self.class_mapping=stl10_map
        elif dataset=='imageclef':
            print('Using image-clef dataset')
            self.class_mapping=imageclef_map
        elif dataset=='fmnist':
            print('Using fmnist dataset')
            self.class_mapping=fmnist_map
        elif dataset=='tign':
            print('Using tiny imagenet dataset')
            self.class_mapping=tign_map
        elif dataset=='mign':
            print('Using mini imagenet dataset')
            self.class_mapping=mign_map
        elif dataset=='office31':
            print('Using office 31 dataset')
            self.class_mapping=office31_map
        else:
            print('Using Office home dataset')
            self.class_mapping=officehome_map
            
        self.__gen_file_list()
        self.__preprocess_graphs()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.files)

    def __getitem__(self, index):
        'Generate one example of data'

        features=self.sg_features[index]
        label=self.sg_labels[index]
        spp_map=self.grid_mapping[index]
        
        if self.cache:
            graph=self.sg_graphs[index]
            F_matrix=np.copy(features[0])
            #self.__recenter(F_matrix,absolute=True)
            graph.ndata['h']=torch.from_numpy(F_matrix)
            
        else:        
            adj_matrix=self.adj_matrices[index]        

            if self.data_augment:
                new_adj,new_F,spp_map=self.__apply_da(adj_matrix,features)
            else:
                new_adj=adj_matrix
                new_F=features[0]
                spp_map=self.__compute_spp_map(new_F,self.grid)
                #self.__recenter(new_F,absolute=True)

            graph=self.__create_graph(new_adj)
            graph.ndata['h']=torch.from_numpy(new_F)

        graph.ndata['x']=torch.from_numpy(spp_map)
        return graph,label

    def __swap(self,M):
        flip=random.randint(0,1)
        flip=self.flip_pp
        if flip:
            M[:,[15,9]]=M[:,[9,15]]
            M[:,[16,10]]=M[:,[10,16]]
            M[:,[17,11]]=M[:,[11,17]]
            M[:,[18,12]]=M[:,[12,18]]
            M[:,[19,13]]=M[:,[13,19]]
            M[:,[20,14]]=M[:,[14,20]]
            M[:,[24,21]]=M[:,[21,24]]
            M[:,[25,22]]=M[:,[22,25]]
            M[:,[26,23]]=M[:,[23,26]]
            M[:,[34,31]]=M[:,[31,34]]
            M[:,[35,32]]=M[:,[32,35]]
            M[:,[36,33]]=M[:,[33,36]]
            M[:,[44,41]]=M[:,[41,44]]
            M[:,[45,42]]=M[:,[42,45]]
            M[:,[46,43]]=M[:,[43,46]]
            M[:,[54,51]]=M[:,[51,54]]
            M[:,[55,52]]=M[:,[52,55]]
            M[:,[56,53]]=M[:,[53,56]]
            
    def __read_cemv_file(self,fid):

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

    def __preprocess_adj_numpy(self,adj, symmetric=True):
        adj = adj + np.eye(adj.shape[0])
        adj = self.__normalize_adj_numpy(adj, symmetric)
        return adj

    def __preprocess_adj_numpy_with_identity(self,adj, symmetric=True):
        adj = adj + np.eye(adj.shape[0])
        adj = self.__normalize_adj_numpy(adj, symmetric)
        adj = np.concatenate([np.eye(adj.shape[0]), adj], axis=0)
        return adj

    def __normalize_adj_numpy(self,adj, symmetric=True):
        if symmetric:
            d = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
            a_norm = adj.dot(d).transpose().dot(d)
        else:
            d = np.diag(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
            a_norm = d.dot(adj)
        return a_norm
        
    def __gen_file_list(self):
        self.files=glob.glob(self.directory+'/*.h5')
        self.files.sort()
        
    def __compute_spp_map(self, F_matrix,cells):
        grid=np.linspace(0,self.image_size,cells+1)
        grid_map=defaultdict(list)
        
        for idx in range(F_matrix.shape[0]):
            pts=F_matrix[idx,:]
            xloc=max(np.searchsorted(grid,pts[0])-1,0,0)
            yloc=max(np.searchsorted(grid,pts[1])-1,0,0)
            grid_map[(xloc,yloc)].append(idx)

        grid_cell=np.ones((F_matrix.shape[0],250),dtype=np.int32)*-1

        idx=0
        for key,value in grid_map.items():
            grid_cell[idx,0]=key[0]
            grid_cell[idx,1]=key[1]
            grid_cell[idx,2]=len(value)
            
            col=3
            for ss in value:
                grid_cell[idx,col]=ss
                col+=1

            idx+=1

        return grid_cell

    def __preprocess_graphs(self):

        for fid in tqdm(self.files):
            adj_matrix,features=self.__read_shock_graph(fid)

            obj=os.path.basename(fid)
            if self.dataset=='tign' or self.dataset=='mign':
                class_name=obj[:obj.find('_')]
            else:
                obj=re.split(r'[0-9].*',obj)[0]
                class_name=obj[:obj.rfind('_')]

            label=self.class_mapping[class_name]
            grid_cell=self.__compute_spp_map(features[0],self.grid)

            self.__recenter(features[0],absolute=True)
            
            self.adj_matrices.append(adj_matrix)
            self.sg_labels.append(label)
            self.sg_features.append(features)
            self.grid_mapping.append(grid_cell)
            
            if self.cache:
                graph=self.__create_graph(adj_matrix)
                self.sg_graphs.append(graph)

    def __normalize_features(self,features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features

    def __round_nearest(self,x, a):
        return np.round(np.round(x / a) * a, -int(math.floor(math.log10(a))))

    def __translate_points(self,pt,v,length):
        i=0
        trans_points=np.zeros((pt.shape[0],6))
        for g in range(0,v.shape[1]):
            x=pt[:,0]+length*np.sin(v[:,g])
            y=pt[:,1]+length*np.cos(v[:,g])
            trans_points[:,i]=x
            i=i+1
            trans_points[:,i]=y
            i=i+1
            
        return trans_points

    def __get_pixel_values(self,F_matrix,color_space):

        pixel_values=np.zeros((F_matrix.shape[0],21),dtype=np.float32)

        zero_set=np.array([0.0,0.0])
        zcoord=np.array([0,1,2])
        for f in range(F_matrix.shape[0]):
            sg_xcoord=np.repeat(F_matrix[f,0],3)
            sg_ycoord=np.repeat(F_matrix[f,1],3)
            sg_cs=ndimage.map_coordinates(color_space,[sg_xcoord,sg_ycoord,zcoord])
            pixel_values[f,:3]=sg_cs
            
            
            bp1_xcoord=np.repeat(F_matrix[f,9],3)
            bp1_ycoord=np.repeat(F_matrix[f,10],3)
            bp1_cs=ndimage.map_coordinates(color_space,[bp1_xcoord,bp1_ycoord,zcoord])
            pixel_values[f,3:6]=bp1_cs

            bp1_xcoord=np.repeat(F_matrix[f,15],3)
            bp1_ycoord=np.repeat(F_matrix[f,16],3)
            bp1_cs=ndimage.map_coordinates(color_space,[bp1_xcoord,bp1_ycoord,zcoord])
            pixel_values[f,12:15]=bp1_cs

            if np.array_equal(F_matrix[f,11:13],zero_set)==False:
                bp2_xcoord=np.repeat(F_matrix[f,11],3)
                bp2_ycoord=np.repeat(F_matrix[f,12],3)
                bp2_cs=ndimage.map_coordinates(color_space,[bp2_xcoord,bp2_ycoord,zcoord])
                pixel_values[f,6:9]=bp2_cs

                bp2_xcoord=np.repeat(F_matrix[f,17],3)
                bp2_ycoord=np.repeat(F_matrix[f,18],3)
                bp2_cs=ndimage.map_coordinates(color_space,[bp2_xcoord,bp2_ycoord,zcoord])
                pixel_values[f,15:18]=bp2_cs

            if np.array_equal(F_matrix[f,13:15],zero_set)==False:
                bp3_xcoord=np.repeat(F_matrix[f,13],3)
                bp3_ycoord=np.repeat(F_matrix[f,14],3)
                bp3_cs=ndimage.map_coordinates(color_space,[bp3_xcoord,bp3_ycoord,zcoord])
                pixel_values[f,9:12]=bp3_cs

                bp3_xcoord=np.repeat(F_matrix[f,19],3)
                bp3_ycoord=np.repeat(F_matrix[f,20],3)
                bp3_cs=ndimage.map_coordinates(color_space,[bp3_xcoord,bp3_ycoord,zcoord])
                pixel_values[f,18:21]=bp3_cs

        return pixel_values/255.0
            
    def __apply_da(self,orig_adj_matrix,features):

        F_matrix=np.copy(features[0])
        mask=np.copy(features[1])

        flip=random.randint(0,1)
        if flip:
            F_matrix[:,1]=((self.width-1)-F_matrix[:,1])-self.width/2
            F_matrix[:,3:6]=math.pi-F_matrix[:,3:6]
            
        random_scale=np.random.rand(1)*(2-.5)+.5
        random_scale=random_scale if random_scale != 0 else 1
        random_trans=np.random.randint(-20,20,size=2)
        
        F_matrix[:,:2]+=random_trans
        F_matrix[:,:2]*=random_scale
        F_matrix[:,2]*=random_scale

        v_plus=F_matrix[:,3:6]+F_matrix[:,6:9]
        new_plus=self.__translate_points(F_matrix[:,:2],v_plus,F_matrix[:,2])
        v_minus=F_matrix[:,3:6]-F_matrix[:,6:9]
        new_minus=self.__translate_points(F_matrix[:,:2],v_minus,F_matrix[:,2])
                                             
        F_matrix[:,9:15]=new_plus
        F_matrix[:,15:21]=new_minus

        if flip:
            F_matrix[:,3:6]+=2*math.pi
            F_matrix[:,3]=fixAngle2PiPi_new_vector(F_matrix[:,3])
            F_matrix[:,4]=fixAngle2PiPi_new_vector(F_matrix[:,4])
            F_matrix[:,5]=fixAngle2PiPi_new_vector(F_matrix[:,5])
        
            # plus theta
            F_matrix[:,21] = fixAngleMPiPi_new_vector(F_matrix[:,3]+F_matrix[:,6]-(math.pi/2.0));
            F_matrix[:,22] = fixAngleMPiPi_new_vector(F_matrix[:,4]+F_matrix[:,7]-(math.pi/2.0));
            F_matrix[:,23] = fixAngleMPiPi_new_vector(F_matrix[:,5]+F_matrix[:,8]-(math.pi/2.0));
        
            # minus theta
            F_matrix[:,24] = fixAngleMPiPi_new_vector(F_matrix[:,3]-F_matrix[:,6]+math.pi/2.0);
            F_matrix[:,25] = fixAngleMPiPi_new_vector(F_matrix[:,4]-F_matrix[:,7]+math.pi/2.0);
            F_matrix[:,26] = fixAngleMPiPi_new_vector(F_matrix[:,5]-F_matrix[:,8]+math.pi/2.0);

        #scale arclength, curvature
        F_matrix[:,28:58] *= random_scale
        
        #mask
        F_matrix=F_matrix*mask
        
        self.trans=(flip,random_scale[0],random_trans)
        F_pruned,adj_pruned,mask_pruned=self.__prune_ob(F_matrix,orig_adj_matrix,mask,self.image_size*random_scale)
        new_adj_matrix,new_F_matrix,new_mask=self.__compute_sorted_order(F_pruned,adj_pruned,mask_pruned)

        new_center=np.zeros(2)
        new_center[0]=(self.image_size*random_scale)/2.0
        new_center[1]=(self.image_size*random_scale)/2.0
        factor=((self.image_size*random_scale)/2.0)*1.2

        spp_map=self.__compute_spp_map(new_F_matrix,self.grid)

        self.__recenter(new_F_matrix,new_center,factor,absolute=True)
        
        return new_adj_matrix,new_F_matrix,spp_map

    def __compute_sorted_order(self,F_matrix,orig_adj_matrix,mask):

        node_order={}
        
        # define a sorting order for nodes in graph
        key_tuples=[]
        for i in range(F_matrix.shape[0]):
            key_tuples.append((F_matrix[i][0],
                               F_matrix[i][1],
                               F_matrix[i][2],
                               i))
            
        sorted_tuples=sorted(key_tuples,key=itemgetter(0,1,2),reverse=False)
        new_adj_matrix=np.zeros((orig_adj_matrix.shape[0],orig_adj_matrix.shape[0]))
        
        new_list=[]
        for idx in range(0,len(sorted_tuples)):
            key=sorted_tuples[idx][3]
            value=idx
            node_order[key]=value
            new_list.append(key)
            
 
        for row in range(0,orig_adj_matrix.shape[0]):
            source_idx=node_order[row]
            neighbors=orig_adj_matrix[row,:]
            target=np.nonzero(neighbors)[0]
            for t in target:
                target_idx=node_order[t]
                new_adj_matrix[source_idx][target_idx]=1

        new_F_matrix=F_matrix[new_list,:]
        new_mask=mask[new_list,:]
        return new_adj_matrix,new_F_matrix,new_mask
    
    def __recenter(self,F_matrix,center=None,factor=None,absolute=True):

        if center is None:
            center=self.center

        if factor is None:
            factor=self.factor
            
        # radius of shock point
        if absolute:

            rad_scale=self.norm_factors['rad_scale']
            angle_scale=self.norm_factors['angle_scale']
            length_scale=self.norm_factors['length_scale']
            curve_scale=self.norm_factors['curve_scale']
            poly_scale=self.norm_factors['poly_scale']

            if self.edge_app:
                offset=13
            else:
                offset=10

            # scale shock radius
            F_matrix[:,2] /= rad_scale

            # scale shock curvature
            F_matrix[:,28] /= curve_scale
            F_matrix[:,28+offset] /= curve_scale
            F_matrix[:,28+2*offset] /= curve_scale

            # scale plus curvature
            F_matrix[:,31] /= curve_scale
            F_matrix[:,31+offset] /= curve_scale
            F_matrix[:,31+2*offset] /= curve_scale

            # scale minus curvature
            F_matrix[:,34] /= curve_scale
            F_matrix[:,34+offset] /= curve_scale
            F_matrix[:,34+2*offset] /= curve_scale

            # scale shock length
            F_matrix[:,29] /= length_scale
            F_matrix[:,29+offset] /= length_scale
            F_matrix[:,29+2*offset] /= length_scale

            # scale plus length
            F_matrix[:,32] /= length_scale
            F_matrix[:,32+offset] /= length_scale
            F_matrix[:,32+2*offset] /= length_scale

            # scale minus length
            F_matrix[:,35] /= length_scale
            F_matrix[:,35+offset] /= length_scale
            F_matrix[:,35+2*offset] /= length_scale

            # scale shock angle
            F_matrix[:,30] /= angle_scale
            F_matrix[:,30+offset] /= angle_scale
            F_matrix[:,30+2*offset] /= angle_scale

            # scale plus angle
            F_matrix[:,33] /= angle_scale
            F_matrix[:,33+offset] /= angle_scale
            F_matrix[:,33+2*offset] /= angle_scale

            # scale minus angle
            F_matrix[:,36] /= angle_scale
            F_matrix[:,36+offset] /= angle_scale
            F_matrix[:,36+2*offset] /= angle_scale

            # scale poly scale
            F_matrix[:,37] /= poly_scale
            F_matrix[:,37+offset] /= poly_scale
            F_matrix[:,37+2*offset] /= poly_scale

        else:

            # scale radius
            rad_scale=np.max(F_matrix[:,2])
            F_matrix[:,2] /= rad_scale

            curve_scale=np.max(np.abs(np.concatenate((F_matrix[:,28],F_matrix[:,38],F_matrix[:,48],
                                                      F_matrix[:,31],F_matrix[:,41],F_matrix[:,51],
                                                      F_matrix[:,34],F_matrix[:,44],F_matrix[:,54]),axis=0)))

            # scale shock curvature
            F_matrix[:,28] /= curve_scale
            F_matrix[:,38] /= curve_scale
            F_matrix[:,48] /= curve_scale

            # scale plus curvature
            F_matrix[:,31] /= curve_scale
            F_matrix[:,41] /= curve_scale
            F_matrix[:,51] /= curve_scale

            # scale minus curvature
            F_matrix[:,34] /= curve_scale
            F_matrix[:,44] /= curve_scale
            F_matrix[:,54] /= curve_scale

            length_scale=np.max(np.abs(np.concatenate((F_matrix[:,29],F_matrix[:,39],F_matrix[:,49],
                                                       F_matrix[:,32],F_matrix[:,42],F_matrix[:,52],
                                                       F_matrix[:,35],F_matrix[:,45],F_matrix[:,55]),axis=0)))
            
            # scale shock length
            F_matrix[:,29] /= length_scale
            F_matrix[:,39] /= length_scale
            F_matrix[:,49] /= length_scale

            # scale plus length
            F_matrix[:,32] /= length_scale
            F_matrix[:,42] /= length_scale
            F_matrix[:,52] /= length_scale

            #scale minus length
            F_matrix[:,35] /= length_scale
            F_matrix[:,45] /= length_scale
            F_matrix[:,55] /= length_scale

            angle_scale=np.max(np.abs(np.concatenate((F_matrix[:,30],F_matrix[:,40],F_matrix[:,50],
                                                      F_matrix[:,33],F_matrix[:,43],F_matrix[:,53],
                                                      F_matrix[:,36],F_matrix[:,46],F_matrix[:,56]),axis=0)))


            # scale shock angle
            F_matrix[:,30] /= angle_scale
            F_matrix[:,40] /= angle_scale
            F_matrix[:,50] /= angle_scale

            # scale plus angle
            F_matrix[:,33] /= angle_scale
            F_matrix[:,43] /= angle_scale
            F_matrix[:,53] /= angle_scale

            #scale minus angle
            F_matrix[:,36] /= angle_scale
            F_matrix[:,46] /= angle_scale
            F_matrix[:,56] /= angle_scale

            poly_scale=np.max(np.abs(np.concatenate((F_matrix[:,37],F_matrix[:,47],F_matrix[:,57]),axis=0)))

            F_matrix[:,37] /= poly_scale
            F_matrix[:,47] /= poly_scale
            F_matrix[:,57] /= poly_scale

        # theta of node
        F_matrix[:,3] /= 2.0*math.pi
        F_matrix[:,4] /= 2.0*math.pi
        F_matrix[:,5] /= 2.0*math.pi

        # phi of node
        F_matrix[:,6] /= math.pi
        F_matrix[:,7] /= math.pi
        F_matrix[:,8] /= math.pi

        # plus theta
        F_matrix[:,21] /= math.pi
        F_matrix[:,22] /= math.pi
        F_matrix[:,23] /= math.pi

        # minus theta
        F_matrix[:,24] /= math.pi
        F_matrix[:,25] /= math.pi
        F_matrix[:,26] /= math.pi

        # remove ref pt for contour and shock points
        F_matrix[:,:2] -=center

        zero_set=np.array([0.0,0.0])

        for row_idx in range(0,F_matrix.shape[0]):
            F_matrix[row_idx,9:11]-=center
            F_matrix[row_idx,15:17]-=center
            
            if np.array_equal(F_matrix[row_idx,11:13],zero_set)==False:
                F_matrix[row_idx,11:13]-=center
                F_matrix[row_idx,17:19]-=center
                
            if np.array_equal(F_matrix[row_idx,13:15],zero_set)==False:
                F_matrix[row_idx,13:15]-=center
                F_matrix[row_idx,19:21]-=center

        F_matrix[:,:2] /= factor
        F_matrix[:,9:15] /= factor
        F_matrix[:,15:21] /= factor


    def __prune_ob(self,F_matrix,adj_matrix,mask,scale):

        # first round of delete
        rows=np.unique(np.where((F_matrix[:,:2]<0)|(F_matrix[:,:2]>=scale))[0])
        F_matrix=np.delete(F_matrix,rows,axis=0)
        adj_matrix=np.delete(adj_matrix,rows,axis=0)
        adj_matrix=np.delete(adj_matrix,rows,axis=1)
        mask=np.delete(mask,rows,axis=0)

        # check if any disconnected nodes
        out_degree=np.where(~adj_matrix.any(axis=1))[0]
        in_degree=np.where(~adj_matrix.any(axis=0))[0]
        rows=np.intersect1d(out_degree,in_degree)

        # second round of delete
        if rows.shape[0]:
            F_matrix=np.delete(F_matrix,rows,axis=0)
            adj_matrix=np.delete(adj_matrix,rows,axis=0)
            adj_matrix=np.delete(adj_matrix,rows,axis=1)
            mask=np.delete(mask,rows,axis=0)

        return F_matrix,adj_matrix,mask
        
    def __unwrap_data(self,F_matrix):

        #make a copy
        feature_matrix=F_matrix
        
        # theta of node
        feature_matrix[:,3] *= 2.0*math.pi
        feature_matrix[:,4] *= 2.0*math.pi
        feature_matrix[:,5] *= 2.0*math.pi

        # phi of node
        feature_matrix[:,6] *= math.pi
        feature_matrix[:,7] *= math.pi
        feature_matrix[:,8] *= math.pi

        # plus theta
        feature_matrix[:,21] *= math.pi
        feature_matrix[:,22] *= math.pi
        feature_matrix[:,23] *= math.pi

        # minus theta
        feature_matrix[:,24] *= math.pi
        feature_matrix[:,25] *= math.pi
        feature_matrix[:,26] *= math.pi

        zero_set=np.array([0.0,0.0])

        mask=np.ones(feature_matrix.shape,dtype=np.float32)
        for row_idx in range(0,feature_matrix.shape[0]):
                        
            if np.array_equal(feature_matrix[row_idx,11:13],zero_set)==True:
                mask[row_idx,11:13]=0
                mask[row_idx,4]=0
                mask[row_idx,7]=0
                mask[row_idx,22]=0
                mask[row_idx,25]=0
                mask[row_idx,17:19]=0

            if np.array_equal(feature_matrix[row_idx,13:15],zero_set)==True:
                mask[row_idx,13:15]=0
                mask[row_idx,5]=0
                mask[row_idx,8]=0
                mask[row_idx,23]=0
                mask[row_idx,26]=0
                mask[row_idx,19:21]=0


        return feature_matrix,mask
        
    def __read_shock_graph(self,sg_file):
        fid=h5py.File(sg_file,'r')

        # read cemv file
        # suffix=os.path.basename(sg_file)
        # sdir=os.path.dirname(sg_file)
        # prefix=suffix[:suffix.find('-')]
        # cemv_file=sdir+'/'+prefix+'.cemv'
        # con_points=self.__read_cemv_file(cemv_file)

        # feature_data=fid.get('feature')
        # F_matrix=np.array(feature_data).astype(np.float32)
        # F_matrix=F_matrix[:,:58]
         
        # read in features
        feature_data=fid.get('feature')
        temp=np.array(feature_data).astype(np.float32)
        sg_features=temp[:,:67]

        if self.edge_app==False:
            sg_features=np.delete(sg_features,[38,39,40,51,52,53,64,65,66],axis=1)
        else:
            sg_features[:,38:41] /= 255.0
            sg_features[:,51:54] /= 255.0
            sg_features[:,64:67] /= 255.0
            
        if self.node_app:
            color_features=temp[:,127:]/255.0
            F_matrix=np.concatenate((sg_features,color_features),axis=1)
        else:
            F_matrix=sg_features
        

        # get shape context
        # a = SC()
        # sg_points=np.copy(F_matrix[:,:2])
        # sg_points[np.abs(sg_points)<1.0e-10]=0.0
        # query=list(map(tuple,sg_points))
        # sc_features=a.compute(query,con_points)
        # norm_SC=normalize(sc_features,norm='l1').astype(np.float32)
        # F_matrix=np.concatenate((F_matrix[:,:58],norm_SC),axis=1)

        # sg_features=F_matrix[:,:58]
        # color_features=F_matrix[:,118:139]/255.0
        # F_matrix=np.concatenate((sg_features,color_features),axis=1)

        # orig=F_matrix[:,58:118]
        # flag=np.array_equal(orig,P)
        # if flag == False:
        #     np.savetxt('orig.txt',orig)
        #     np.savetxt('new.txt',P)
        #     exit()
        # print(np.max(np.abs(orig-P)))
        # norm_SC=normalize(SC,norm='l1')
        # F_matrix[:,58:118]=norm_SC
        # F_matrix=F_matrix[:,:118]
        
        # read in adj matrix
        adj_data=fid.get('adj_matrix')
        adj_matrix=np.array(adj_data)

        # read in image dims
        dims=fid.get('dims')
        dims=np.array(dims)

        fid.close()
        
        # remove normalization put in
        F_matrix_unwrapped,mask=self.__unwrap_data(F_matrix)
        
        # center of bounding box
        # will be a constant across
        self.width=(F_matrix_unwrapped[1,1]-F_matrix_unwrapped[0,1])
        self.image_size=dims[0]
        self.center=np.array([self.image_size/2.0,self.image_size/2.0])
        self.factor=(self.image_size/2.0)*1.2

        F_combined_pruned,adj_matrix_pruned,mask_pruned=self.__prune_ob(F_matrix_unwrapped,adj_matrix,mask,self.image_size)
            
        if self.flip_pp:
            F_combined_pruned[:,1]=((self.width-1)-F_combined_pruned[:,1])-self.width/2
            F_combined_pruned[:,3:6]=math.pi-F_combined_pruned[:,3:6]

            v_plus=F_combined_pruned[:,3:6]+F_combined_pruned[:,6:9]
            new_plus=self.__translate_points(F_combined_pruned[:,:2],v_plus,F_combined_pruned[:,2])
            v_minus=F_combined_pruned[:,3:6]-F_combined_pruned[:,6:9]
            new_minus=self.__translate_points(F_combined_pruned[:,:2],v_minus,F_combined_pruned[:,2])

            F_combined_pruned[:,9:15]=new_plus
            F_combined_pruned[:,15:21]=new_minus

            F_combined_pruned[:,3:6]+=2*math.pi
            F_combined_pruned[:,3]=fixAngle2PiPi_new_vector(F_combined_pruned[:,3])
            F_combined_pruned[:,4]=fixAngle2PiPi_new_vector(F_combined_pruned[:,4])
            F_combined_pruned[:,5]=fixAngle2PiPi_new_vector(F_combined_pruned[:,5])

            # plus theta
            F_combined_pruned[:,21] = fixAngleMPiPi_new_vector(F_combined_pruned[:,3]+F_combined_pruned[:,6]-(math.pi/2.0));
            F_combined_pruned[:,22] = fixAngleMPiPi_new_vector(F_combined_pruned[:,4]+F_combined_pruned[:,7]-(math.pi/2.0));
            F_combined_pruned[:,23] = fixAngleMPiPi_new_vector(F_combined_pruned[:,5]+F_combined_pruned[:,8]-(math.pi/2.0));

            # minus theta
            F_combined_pruned[:,24] = fixAngleMPiPi_new_vector(F_combined_pruned[:,3]-F_combined_pruned[:,6]+math.pi/2.0);
            F_combined_pruned[:,25] = fixAngleMPiPi_new_vector(F_combined_pruned[:,4]-F_combined_pruned[:,7]+math.pi/2.0);
            F_combined_pruned[:,26] = fixAngleMPiPi_new_vector(F_combined_pruned[:,5]-F_combined_pruned[:,8]+math.pi/2.0);

            F_combined_pruned=F_combined_pruned*mask_pruned

        # resort to be safe
        new_adj_matrix,new_F_matrix,new_mask=self.__compute_sorted_order(F_combined_pruned,adj_matrix_pruned,mask_pruned)
        
        return new_adj_matrix,(new_F_matrix,new_mask)

    def __create_graph(self,adj_matrix):
    
        # convert to dgl
        G=dgl.DGLGraph()
        G.add_nodes(adj_matrix.shape[0])
        G.set_n_initializer(dgl.init.zero_initializer)
 
        for row in range(0,adj_matrix.shape[0]):
            neighbors=adj_matrix[row,:]
            target=np.nonzero(neighbors)[0]
            source=np.zeros(target.shape)+row

            G.add_edges(row,row)
            
            if target.size:
                G.add_edges(source,target)
                if self.symmetric:
                    G.add_edges(target,source)
                                
        return G
     
def collate(samples,device_name):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels).to(device_name)
