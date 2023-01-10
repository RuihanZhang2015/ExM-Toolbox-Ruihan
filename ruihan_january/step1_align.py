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

fov_code_pairs = [[fov,code] for fov in [16] for code in [0,1,2,3,4,5,6]]
# exseq.transform_405_acceleration(fov_code_pairs,num_cpu=1)
# exseq.transform_others_acceleration(fov_code_pairs,num_cpu = 1)
# exseq.inspect_alignment_multiFovCode(fov_code_pairs,num_layer=4)

# /mp/nas3/ruihan/20221218_zebrafish/processed/code1/

# (CenterOfRotationPoint 405.437988 405.437988 198.000000)
# (CompressResultImage "false")
# (ComputeZYX "false")
# (DefaultPixelValue 0.000000)
# (Direction 1.000000 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 1.000000)
# (FinalBSplineInterpolationOrder 1.000000)
# (FixedImageDimension 3.000000)
# (FixedInternalImagePixelType "float")
# (HowToCombineTransforms "Compose")
# (Index 0.000000 0.000000 0.000000)
# (InitialTransformParametersFileName "NoInitialTransform")
# (MovingImageDimension 3.000000)
# (MovingInternalImagePixelType "float")
# (NumberOfParameters 6.000000)
# (Origin 0.000000 0.000000 0.000000)
# (ResampleInterpolator "FinalBSplineInterpolator")
# (Resampler "DefaultResampler")
# (ResultImageFormat "nii")
# (ResultImagePixelType "float")
# (Size 500.000000 500.000000 100.000000)
# (Spacing 1.625000 1.625000 4.000000)
# (Transform "EulerTransform")
# (TransformParameters -0.206903 -0.123118 -0.060615 -104.648003 159.065002 -94.909500)
# (UseDirectionCosines "true")
     
# (CenterOfRotationPoint 405.437988 405.437988 398.000000)
# (CompressResultImage "false")
# (ComputeZYX "false")
# (DefaultPixelValue 0.000000)
# (Direction 1.000000 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 1.000000)
# (FinalBSplineInterpolationOrder 1.000000)
# (FixedImageDimension 3.000000)
# (FixedInternalImagePixelType "float")
# (HowToCombineTransforms "Compose")
# (Index 0.000000 0.000000 0.000000)
# (InitialTransformParametersFileName "NoInitialTransform")
# (MovingImageDimension 3.000000)
# (MovingInternalImagePixelType "float")
# (NumberOfParameters 6.000000)
# (Origin 0.000000 0.000000 0.000000)
# (ResampleInterpolator "FinalBSplineInterpolator")
# (Resampler "DefaultResampler")
# (ResultImageFormat "nii")
# (ResultImagePixelType "float")
# (Size 500.000000 500.000000 200.000000)
# (Spacing 1.625000 1.625000 4.000000)
# (Transform "EulerTransform")
# (TransformParameters -0.145646 -0.142664 -0.088569 -85.853203 128.617004 -50.458500)
# (UseDirectionCosines "true")