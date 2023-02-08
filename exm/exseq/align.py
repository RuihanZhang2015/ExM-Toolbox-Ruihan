# from multiprocessing import current_process
# import queue # imported for using queue.Empty exception
# import h5py
# import time
# import cupy as cp
# import numpy as np
# import os
# import multiprocessing
# from multiprocessing import Process,Queue
# import pickle
# import collections

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import h5py
from exm.io.io import nd2ToVol
from hijack import *












def align405_single_acceleration(self,tasks_queue,q_lock):
    
    while True: # Check for remaining task in the Queue

        try:
            with q_lock:
                fov,code = tasks_queue.get_nowait()
                print('Remaining tasks to process : {}'.format(tasks_queue.qsize()))
        except queue.Empty:
            print("No task left for "+ current_process().name)
            break

        else:

            if code == self.args.ref_code:
                        
                fix_vol = nd2ToVol(self.args.fix_path, fov)
                with h5py.File(self.args.out_dir + 'code{}/{}.h5'.format(self.args.ref_code,fov), 'w') as f:
                    f.create_dataset('405', fix_vol.shape, compression="gzip", dtype=fix_vol.dtype, data = fix_vol)

            else:
                print("Code {} FOV {} Started on {}".format(code,fov,current_process().name))
                cfg = load_cfg()
                align = alignBuild(cfg)
                align.buildSitkTile()

                with h5py.File(self.args.h5_path.format(self.args.ref_code,fov), 'r+') as f:
                    fix_vol = f['405'][:]
                mov_vol = nd2ToVol(self.args.mov_path.format(code,'405',4), fov)
        
                z_nums = mov_vol.shape[0]

                # lazy exception due to SITK failing sometimes
                try:
                    tform = align.computeTransformMap(fix_vol, mov_vol)
                    result = align.warpVolume(mov_vol, tform)
                    print(align.__dict__)

                    with h5py.File(self.args.h5_path.format(code,fov), 'w') as f:
                        f.create_dataset('405', result.shape, dtype=result.dtype, data = result)

                    align.writeTransformMap(self.args.out_dir + 'code{}/tforms/{}.txt'.format(code,fov), tform)

                    print("Code {} FOV {} on {} Done".format(code,fov,current_process().name))    

                except Exception as e:
                    print(e)
                    # write code, fov to .txt file if it doesn't work
                    with open(self.args.out_dir + '/failed.txt','a') as f:
                        f.write(f'code{code},fov{fov}\n')
                    print('code{},fov{} failed'.format(code,fov))

            try:
                os.system('chmod -R 777 {}  >/dev/null 2>&1'.format(self.args.work_path))
                os.system('chmod -R 777 {}  >/dev/null 2>&1'.format(self.args.out_path))
            except:
                pass


def transform_405_lowres(self,fov_code_pairs,num_cpu=None, modified= False):


        
    '''
    exseq.transform_405_acceleration(fov_code_pairs,num_cpu=2)
    '''
        
       
    os.environ["OMP_NUM_THREADS"] = "1"


    # Use a quarter of the available CPU resources to finish the tasks; you can increase this if the server is accessible for this task only.
    if num_cpu == None:
        cpu_execution_core = multiprocessing.cpu_count() / 4
    else:
        cpu_execution_core = num_cpu
    # List to hold the child processes.
    child_processes = [] 
    # Queue to hold all the puncta extraction tasks.
    tasks_queue = Queue() 
    # Queue lock to avoid race condition.
    q_lock = multiprocessing.Lock()
    # Get the extraction tasks starting time. 
        
    start_time = time.time()
        
    # # Clear the child processes list.
    child_processes = [] 

    # Add all the align405 to the queue.
    for fov,code in fov_code_pairs:
        tasks_queue.put((fov,code))

    for w in range(int(cpu_execution_core)):
        p = Process(target=align405_single_acceleration, args=(tasks_queue,q_lock))
        child_processes.append(p)
        p.start()

    for p in child_processes:
        p.join()

    with open(self.args.out_dir + '/process_time.txt','a') as f:
        f.write(f'Align405_other_round_time,{str(time.time()-start_time)} s\n')
   