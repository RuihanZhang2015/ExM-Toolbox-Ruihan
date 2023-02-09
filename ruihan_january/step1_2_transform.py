import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import h5py
from exm.io.io import nd2ToVol
from hijack import *

ref_code = 0
mov_path = '/mp/nas3/ruihan/20221218_zebrafish/code{}/Channel{} SD_Seq000{}.nd2'
out_dir = '/mp/nas3/ruihan/20221218_zebrafish/processed/'

code_fov_pairs = [[code,fov] for code in range(1,7) for fov in range(30)]


for code,fov in code_fov_pairs:

    if tuple([code,fov]) not in starting:
        continue

    print(code,fov,'----------------------')

    h5_name = '/mp/nas3/ruihan/20221218_zebrafish/processed/code{}/{}.h5'.format(code,fov)
    fix_start,mov_start,last = starting[tuple([code,fov])]

    with h5py.File(h5_name, 'r+') as f:
        mov_vol = f['405'][:,:,:]

    mov_vol_sitk = sitk.GetImageFromArray(mov_vol)
    mov_vol_sitk.SetSpacing([1.625,1.625,4.0])

    ### Apply transform on full res
    transform_map = sitk.ReadParameterFile('/mp/nas3/ruihan/20221218_zebrafish/processed/code{}/tforms/{}.txt'.format(code,fov))
    print(transform_map)

    # Size
    transform_map["Size"] = tuple([str(x) for x in mov_vol.shape[::-1]])

    # Transform
    trans_um = np.array([float(x) for x in transform_map["TransformParameters"]])
    trans_um[-1] -= (fix_start-mov_start)*4
    transform_map["TransformParameters"] = tuple([str(x) for x in trans_um])     

    # center of rotation
    cen_um = np.array([float(x) for x in transform_map['CenterOfRotationPoint']])   
    cen_um[-1] += mov_start*4
    transform_map['CenterOfRotationPoint'] = tuple([str(x) for x in cen_um])  

    transformix = sitk.TransformixImageFilter()
    transformix.SetTransformParameterMap(transform_map)
    transformix.SetMovingImage(mov_vol_sitk)
    transformix.Execute()
    out = sitk.GetArrayFromImage(transformix.GetResultImage())

    with h5py.File('/mp/nas3/ruihan/20221218_zebrafish/processed/code{}/{}_transformed.h5'.format(code,fov), 'w') as f:
        f.create_dataset('405', out.shape, dtype=out.dtype, data = out)
        
import os
os.system("curl -X POST -H \'Content-type: application/json\' --data \'{\"text\":\"full resolution fov7 code 1 finished!\"}\' https://hooks.slack.com/services/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")
