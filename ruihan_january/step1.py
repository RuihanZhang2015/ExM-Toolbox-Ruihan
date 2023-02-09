import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import h5py
from exm.io.io import nd2ToVol
from hijack import *
import os

args = Args(
    mov_path = '/mp/nas3/ruihan/20221218_zebrafish/code{}/Channel{} SD_Seq000{}.nd2',
    layout_file = '/mp/nas3/ruihan/20221218_zebrafish/code0/out.txt',
    out_path = '/mp/nas3/ruihan/20221218_zebrafish/',
    sheet_path = '/mp/nas2/ruihan/ExM-Toolbox/ruihan_september/gene_list.numbers',
    codes = [0,1,2,3,4,5,6],
    ref_code = 0,
    mapping = False,
    fovs = range(30))
exseq = ExSeq(args)

blacklist = [(12,1),(17,1),(10,6),(24,6)]
fov_code_pairs = [[fov,code] for fov in range(30) for code in [0] if tuple([fov,code]) not in blacklist]
exseq.transform_others_highres(fov_code_pairs,num_cpu=2)
