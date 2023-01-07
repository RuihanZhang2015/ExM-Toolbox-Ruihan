from exm.exseq.args import Args
from exm.exseq.exseq import ExSeq
args = Args(
    mov_path = '/mp/nas3/ruihan/20221218_zebrafish/code{}/Channel{} SD_Seq000{}.nd2',
    layout_file = '/mp/nas3/ruihan/20221218_zebrafish/code0/out.csv',
    out_path = '/mp/nas3/ruihan/20221218_zebrafish/',
    sheet_path = '/mp/nas2/ruihan/ExM-Toolbox/ruihan_september/gene_list.numbers',
    codes = [0,1,2,3,4,5,6],
    ref_code = 0,
    mapping = False,
    fovs = None)
exseq = ExSeq(args)

fov_code_pairs = [[fov,code] for fov in [4] for code in range(1,7)]
exseq.transform_405_acceleration(fov_code_pairs,num_cpu=1)
exseq.transform_others_acceleration(fov_code_pairs,num_cpu = 1)
exseq.inspect_alignment_multiFovCode(fov_code_pairs,num_layer=4)

