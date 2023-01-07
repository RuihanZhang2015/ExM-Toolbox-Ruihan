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


code = 1
fov = 4
with open(exseq.args.out_dir + 'code{}/tforms/{}.txt'.format(code,fov),'r') as f:
    lines = f.readlines()

lines[0] = '(CenterOfRotationPoint 1664.00000 1664.00000 398.000000)\n' # TBD

print(lines[19] )
with open(exseq.args.out_dir + 'code{}/tforms/{}_hijack.txt'.format(code,fov),'w') as f:
    for line in lines:
        f.writelines(line)
print(exseq.args.out_dir + 'code{}/tforms/{}_hijack.txt'.format(code,fov))
