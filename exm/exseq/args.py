import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from ..config.utils import load_cfg
from ..align.build import alignBuild
from numbers_parser import Document
from ..io.io import createFolderStruc
from scipy.spatial.distance import cdist
from ..io.io import nd2ToVol
from nd2reader import ND2Reader
import plotly.graph_objects as go
import plotly.express as px
import time
import multiprocessing
from multiprocessing import Process,Queue
import collections
import time

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import pandas as pd
pd.set_option('display.expand_frame_repr', False)

import h5py
import pickle
import cupy as cp
import numpy as np
from tqdm import tqdm
from multiprocessing import current_process
import queue # imported for using queue.Empty exception


class Args():
    
    def __init__(self,
                mov_path = '/mp/nas3/ruihan/20220916_zebrafish/code{}/Channel{} SD_Seq000{}.nd2',
                layout_file = '/mp/nas3/ruihan/20220916_zebrafish/code0/out.csv',
                out_path = '/mp/nas3/ruihan/20220916_zebrafish/',
                sheet_path = 'gene_list.numbers',
                codes = [0,1,2,5],
                ref_code = 0,
                gene_list = 'gene_list.numbers',
                mapping = False,
                fovs = None):
        
        self.ref_code = ref_code
        self.mov_path = mov_path
        self.fix_path = self.mov_path.format(self.ref_code,405,4)
        
        self.out_path = out_path
        self.out_dir = self.out_path + '/processed/'
        self.h5_path = self.out_dir + '/code{}/{}.h5'
        self.work_path = self.out_path + '/puncta/'
        
        self.layout_file = layout_file
        
#         self.ROI_min_default = [0,0,0]
#         self.ROI_max_default = [ND2Reader(args.fix_path).metadata['z_levels'][-1]+1,2048,2048]
        
        if not fovs:
            self.fovs = list(ND2Reader(self.fix_path).metadata['fields_of_view'])
            print('fovs',len(self.fovs))
        else:
            self.fovs = fovs
            
        self.codes = codes
        
        self.code2num = {'a':'0','c':'1','g':'2','t':'3'}
        self.colors = ['red','yellow','green','blue']
        self.colorscales = ['Reds','Oranges','Greens','Blues']
        self.channel_names = ['640','594','561','488','405']
        self.thresholds = [200,300,300,200]
    
        self.sheet_path = gene_list
        doc = Document(sheet_path)
        sheets = doc.sheets()
        tables = sheets[0].tables()
        data = tables[0].rows(values_only=True)

        df = pd.DataFrame(data[1:], columns=data[0])

        
        self.map_gene = collections.defaultdict(list)
        for i in range(len(df)):
            temp = df.loc[i,'Barcode']
            temp = ''.join([self.code2num[temp[code]] for code in self.codes])
            self.map_gene[temp] = df.loc[i,'Gene']

        colors = sns.color_palette(None, len(self.map_gene.keys()))
        self.map_color = {a:b for a,b in zip(self.map_gene.keys(),colors)}
       
   