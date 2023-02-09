import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import h5py
from exm.io.io import nd2ToVol
from hijack import *
import os


def transform_405_acceleration(fov_code_pairs, starting, num_cpu=None):

    import multiprocessing
    import time

    def align405_single_acceleration(tasks_queue,q_lock):

        while True: # Check for remaining task in the Queue
            try:
                with q_lock:
                    fov,code = tasks_queue.get_nowait()
                    print('Remaining tasks to process : {}'.format(tasks_queue.qsize()))
            except multiprocessing.queue.Empty:
                print("No task left for "+ multiprocessing.current_process().name)
                break

            else:
                if code == ref_code:

                    h5_name = '/mp/nas3/ruihan/20221218_zebrafish//processed/code{}/{}.h5'.format(ref_code,fov)
                    if not os.path.exists(h5_name):
                        fix_vol = nd2ToVol(mov_path.format(ref_code,'405',4), fov)
                        with h5py.File(h5_name, 'w') as f:
                            f.create_dataset('405', fix_vol.shape, dtype=fix_vol.dtype, data = fix_vol)

                else:
                    if tuple([code,fov]) not in starting:
                        continue

                    print("Code {} FOV {} Started on {}".format(code,fov,multiprocessing.current_process().name))
                    fix_start,mov_start,last = starting[tuple([code,fov])]

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
                    # elastixImageFilter.LogToConsoleOff()
                    elastixImageFilter.Execute()

                    transform_map = elastixImageFilter.GetTransformParameterMap()
                    sitk.PrintParameterMap(transform_map)
                    sitk.WriteParameterFile(transform_map[0], '/mp/nas3/ruihan/20221218_zebrafish//processed/code{}/tforms/{}.txt'.format(code,fov))

                    ### Apply transform

                    transform_map = sitk.ReadParameterFile('/mp/nas3/ruihan/20221218_zebrafish//processed/code{}/tforms/{}.txt'.format(code,fov))
                    transformix = sitk.TransformixImageFilter()
                    transformix.SetTransformParameterMap(transform_map)
                    transformix.SetMovingImage(mov_vol_sitk)
                    transformix.Execute()
                    out = sitk.GetArrayFromImage(transformix.GetResultImage())

                    with h5py.File('/mp/nas3/ruihan/20221218_zebrafish//processed/code{}/{}_transformed.h5'.format(code,fov), 'w') as f:
                        f.create_dataset('405', out.shape, dtype=out.dtype, data = out)


    os.environ["OMP_NUM_THREADS"] = "1"
    # Use a quarter of the available CPU resources to finish the tasks; you can increase this if the server is accessible for this task only.
    cpu_execution_core = num_cpu if num_cpu else multiprocessing.cpu_count() / 4

    # List to hold the child processes.
    child_processes = []

    # Queue to hold all the puncta extraction tasks.
    tasks_queue = multiprocessing.Queue()

    # Queue lock to avoid race condition.
    q_lock = multiprocessing.Lock()

    # Get the extraction tasks starting time.
    start_time = time.time()


    # Add all the align405 to the queue.
    for fov,code in fov_code_pairs:
        tasks_queue.put((fov,code))

    for w in range(int(cpu_execution_core)):
        p = multiprocessing.Process(target=align405_single_acceleration, args=(tasks_queue,q_lock))
        child_processes.append(p)
        p.start()

    for p in child_processes:
        p.join()

    with open(out_dir + '/process_time.txt','a') as f:
        f.write(f'Align405_other_round_time,{str(time.time()-start_time)} s\n')

sitk.ProcessObject_SetGlobalWarningDisplay(False)
ref_code = 0
mov_path = '/mp/nas3/ruihan/20221218_zebrafish/code{}/Channel{} SD_Seq000{}.nd2'
out_dir = '/mp/nas3/ruihan/20221218_zebrafish/processed/'

fov_code_pairs = [[fov,code] for code in range(2,7)[::-1] for fov in range(30)]
transform_405_acceleration(fov_code_pairs, starting, num_cpu=3)