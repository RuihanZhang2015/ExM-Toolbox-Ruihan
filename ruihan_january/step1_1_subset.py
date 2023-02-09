import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import h5py
from exm.io.io import nd2ToVol
from hijack import *
import os

ref_code = 0
mov_path = '/mp/nas3/ruihan/20221218_zebrafish/code{}/Channel{} SD_Seq000{}.nd2'
out_dir = '/mp/nas3/ruihan/20221218_zebrafish/processed/'
sitk.ProcessObject_SetGlobalWarningDisplay(False)
for code in [3]:
    for fov in range(20,30)[::-1]:

        if tuple([code,fov]) not in starting or tuple([code,fov]) in [(6,0),(6,1)]:
            continue

        print(code,fov)
        fix_start,mov_start,last = starting[tuple([code,fov])]

        ## fix volume
        h5_name = '/mp/nas3/ruihan/20221218_zebrafish//processed/code{}/{}.h5'.format(ref_code,fov)
        if os.path.exists(h5_name):
            with h5py.File(h5_name, 'r+') as f:
                fix_vol = f['405'][fix_start:fix_start+last,:,:]
        else:
            fix_vol = nd2ToVol(mov_path.format(ref_code,'405',4), fov)
            with h5py.File(h5_name, 'w') as f:
                f.create_dataset('405', fix_vol.shape, dtype=fix_vol.dtype, data = fix_vol)
            fix_vol = fix_vol[fix_start:fix_start+last,:,:]

        ## mov volume
        h5_name = '/mp/nas3/ruihan/20221218_zebrafish//processed/code{}/{}.h5'.format(code,fov)
        if os.path.exists(h5_name):
            with h5py.File(out_dir + 'code{}/{}.h5'.format(code,fov), 'r+') as f:
                mov_vol = f['405'][mov_start:mov_start+last,:,:]
        else:
            mov_vol = nd2ToVol(mov_path.format(code,'405',4), fov)
            with h5py.File(h5_name, 'w') as f:
                f.create_dataset('405', mov_vol.shape, dtype=mov_vol.dtype, data = mov_vol)
            mov_vol = mov_vol[mov_start:mov_start+last,:,:]

        ## Align

        elastixImageFilter = sitk.ElastixImageFilter()

        fix_vol_sitk = sitk.GetImageFromArray(fix_vol)
        fix_vol_sitk.SetSpacing([1.625,1.625,4.0])
        elastixImageFilter.SetFixedImage(fix_vol_sitk)

        mov_vol_sitk = sitk.GetImageFromArray(mov_vol)
        mov_vol_sitk.SetSpacing([1.625,1.625,4.0])
        elastixImageFilter.SetMovingImage(mov_vol_sitk)

        parameter_map = sitk.GetDefaultParameterMap('rigid')
        parameter_map['NumberOfSamplesForExactGradient'] = ['1000']  # NumberOfSamplesForExactGradient
        parameter_map['MaximumNumberOfIterations'] = ['15000'] # MaximumNumberOfIterations
        parameter_map['MaximumNumberOfSamplingAttempts'] = ['100'] # MaximumNumberOfSamplingAttempts
        parameter_map['FinalBSplineInterpolationOrder'] = ['1'] #FinalBSplineInterpolationOrder
        parameter_map['NumberOfResolutions'] = ['2']
        elastixImageFilter.SetParameterMap(parameter_map)
        elastixImageFilter.LogToConsoleOff()
        elastixImageFilter.Execute()

        transform_map = elastixImageFilter.GetTransformParameterMap()
        sitk.PrintParameterMap(transform_map)
        sitk.WriteParameterFile(transform_map[0], '/mp/nas3/ruihan/20221218_zebrafish//processed/code{}/tforms/{}.txt'.format(code,fov))

        ### Apply transform

        transform_map = sitk.ReadParameterFile('/mp/nas3/ruihan/20221218_zebrafish//processed/code{}/tforms/{}.txt'.format(code,fov))
        transformix = sitk.TransformixImageFilter()
        transformix.LogToConsoleOff()
        transformix.SetTransformParameterMap(transform_map)

        mov_vol_sitk = mov_vol_sitk[:,:,:100]

        transformix.SetMovingImage(mov_vol_sitk)
        transformix.Execute()
        out = sitk.GetArrayFromImage(transformix.GetResultImage())

        with h5py.File('/mp/nas3/ruihan/20221218_zebrafish//processed/code{}/{}_transformed.h5'.format(code,fov), 'w') as f:
            f.create_dataset('405', out.shape, dtype=out.dtype, data = out)