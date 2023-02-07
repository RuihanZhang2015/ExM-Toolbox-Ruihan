from exm.exseq.args import Args
from exm.exseq.exseq import ExSeq
args = Args(
    mov_path = '/mp/nas3/ruihan/20221218_zebrafish/code{}/Channel{} SD_Seq000{}.nd2',
    layout_file = '/mp/nas3/ruihan/20221218_zebrafish/code0/out.txt',
    out_path = '/mp/nas3/ruihan/20221218_zebrafish/',
    sheet_path = '/mp/nas2/ruihan/ExM-Toolbox/ruihan_september/gene_list.numbers',
    codes = [0,1,2,3,4,5,6],
    ref_code = 0,
    mapping = False,
    fovs = None)
exseq = ExSeq(args)
# exseq.visualize_progress()


fov_code_pairs = [[fov,code] for fov in range(30) for code in range(7)]
exseq.extract(fov_code_pairs,use_gpu=False,num_cpu = 1)
 