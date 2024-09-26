import argparse
import logging
import os
import sys
import nibabel as nib
import numpy as np
import matplotlib
from copy import deepcopy
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from spinalcordtoolbox.image import Image, add_suffix
from scipy.interpolate import InterpolatedUnivariateSpline
from spinalcordtoolbox.registration.landmarks import getRigidTransformFromLandmarks
from spinalcordtoolbox.centerline.core import get_centerline 
from spinalcordtoolbox.resampling import resample_nib
from spinalcordtoolbox.scripts import sct_apply_transfo
from spinalcordtoolbox.scripts import sct_register_multimodal
from spinalcordtoolbox.registration.algorithms import Paramreg, ParamregMultiStep
from spinalcordtoolbox.registration.core import register_wrapper

try:
    # Attempt to import as if it is a part of a package
    from .mm_util import check_image_exists, get_template_paths, validate_register_to_template_args, create_output_dir
except ImportError:
    # Fallback to direct import if run as a standalone script
    from mm_util import check_image_exists, get_template_paths, validate_register_to_template_args, create_output_dir

#naming not functional
# get_parser: parses command line arguments, sets up a) required (image, body region), and b) optional arguments (model, output file name, output directory)
def get_parser():
    parser = argparse.ArgumentParser(
        description="Register an image and segmentations to the specified template.")
    
    # Required arguments
    required = parser.add_argument_group("Required")
    
    required.add_argument("-i", '--input_image', required=True, type=str,
                          help="Input image to segment.")
    
    required.add_argument("-s", '--segmentation_image', required=True, type=str, 
                          help="Segmentation image.")
    
    required.add_argument("-r", '--region', required=True, type=str,
                          help="Anatomical region to segment. Supported regions: abdomen")
    
    # Optional arguments
    optional = parser.add_argument_group("Optional")
    optional.add_argument("-p", '--parameters', required=False, type=str, default='step=1,type=seg,algo=translation,metric=MeanSquares:step=2,type=seg,algo=rigid,metric=MeanSquares:step=3,type=seg,algo=affine,metric=MeanSquares:step=4,type=seg,algo=bsplinesyn,metric=MeanSquares:step=5,type=seg,algo=bsplinesyn,metric=MeanSquares,slicewise=1',
                            help="sct_register_multimodal parameters.")
    
    optional.add_argument("-o", '--output_dir', required=False, type=str,
                            help="Output directory to save the results. If left empty, saves to current working directory.")
    
    return parser

# main: sets up logging, parses command-line arguments using parser, runs model, inference, post-processing
def main():
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    args = parser.parse_args()
    #validate_register_to_template_args(args)

    logging.info(f"Input image: {args.input_image}")
    logging.info(f"Segmentation image: {args.segmentation_image}")
    logging.info(f"Region: {args.region}")

    output_dir=create_output_dir(args.output_dir)
    
    script_path = os.path.abspath(__file__)
    print(f"The absolute path of the script is: {script_path}")

    template_path, template_segmentation_path = get_template_paths('abdomen')

    #Output directory is CWD if None else create specified output directory
    if args.output_dir is None:
        output_dir = os.getcwd()
    elif not os.path.exists(args.output_dir):
        output_dir = os.path.abspath(args.output_dir) 
        os.makedirs(output_dir)
    elif os.path.exists(args.output_dir):
       output_dir=args.output_dir
    else:
        logging.error(f"Error: {args.output_dir}. Output must be path to output directory.")
        sys.exit(1)

      
    #Load template and plot images
    try:
        template = Image(template_path).change_orientation("RPI")
        #plt.imshow(template.data[:,:,template.data.shape[2]//2])
        #plt.show()
    except ImportError:
        print(f"Unable to load {args.region}_template.nii.gz")

    try:
        template_seg = Image(template_segmentation_path).change_orientation("RPI")
        #plt.imshow(template_seg.data[:,:,template_seg.data.shape[2]//2])
        #plt.show()       
    except ImportError:
        print(f"Unable to load {args.region}_template_dseg.nii.gz")
    
    try:
        image_filename = args.input_image
        image = Image(image_filename).change_orientation("RPI")
       # image = resample_nib(image, new_size=[0.5,0.5,0.5], new_size_type='mm', interpolation='linear', mode='nearest')
       # image_filename = add_suffix(image_filename, '_resampled')
       # image.save(os.path.join(output_dir, image_filename), mutable=True)
       # plt.imshow(image.data[:,:,image.data.shape[2]//2])
        #plt.show()
    except ImportError:
        print(f"Unable to load {args.input_image}")

    try:
        image_seg_filename = args.segmentation_image
        image_seg = Image(image_seg_filename).change_orientation("RPI")
        #image_seg = resample_nib(image_seg, new_size=[0.5,0.5,0.5], new_size_type='mm', interpolation='nn', mode='nearest')
        #image_seg_filename = add_suffix(image_seg_filename, '_resampled')
        #image_seg.save(os.path.join(output_dir, image_seg_filename), mutable=True)
        #plt.imshow(image_seg.data[:,:,image_seg.data.shape[2]//2])
        #plt.show()
    except ImportError:
        print(f"Unable to load {args.segmentation_image}")
    
    #for label in np.unique(image_seg.data)[0]:
    for label in np.arange(1,2):

        if label > 0: #iterates through muscles 
            
            template_label_mask = template_seg.data == label        
            image_label_mask = image_seg.data == label #mask is when component is labelled label (0 thru whatever), has dimensions of INPUT
            #If no mask, then assign nan
            if image_label_mask.sum() == 0 or template_label_mask.sum() == 0:
                continue
            else:
                template_label = template_seg.copy()
                template_label.data = template_label_mask * 1
                template_label_filename = add_suffix(template_path, '_seg_label-' + str(int(label)))
                template_label.save(os.path.join(output_dir, template_label_filename), mutable=True)
                                
                image_label = image_seg.copy()
                image_label.data = image_label_mask * 1
                image_label_filename = add_suffix(image_filename, '_seg_label-' + str(int(label)))
                image_label.save(os.path.join(output_dir, image_label_filename), mutable=True)

            #Get and plot centerlines
            image_label_centerline_pix, image_label_arr_ctl_pix, image_label_arr_ctl_der_pix, image_label_fit_results_pix = get_centerline(image_label, space='pix')
            image_label_centerline_pix.save(path=add_suffix(image_label_filename, "_desc-centerline"), mutable=True)
            #ax = plt.axes(projection='3d')
            #ax.plot3D(image_label_arr_ctl_pix[0], image_label_arr_ctl_pix[1], image_label_arr_ctl_pix[2])
            #plt.show()

            template_label_centerline_pix, template_label_arr_ctl_pix, template_label_arr_ctl_der_pix, template_label_fit_results_pix = get_centerline(template_label, space='pix')
            template_label_centerline_pix.save(path=add_suffix(template_label_filename, "_desc-centerline"), mutable=True)
            #ax = plt.axes(projection='3d')
            #ax.plot3D(template_label_arr_ctl_pix[0], template_label_arr_ctl_pix[1], template_label_arr_ctl_pix[2])
            #plt.show()

            image_label_centerline_phys, image_label_arr_ctl_phys, image_label_arr_ctl_der_phys, image_label_fit_results_phys = get_centerline(image_label, space='phys')
            template_label_centerline_phys, template_label_arr_ctl_phys, template_label_arr_ctl_der_phys, template_label_fit_results_phys = get_centerline(template_label, space='phys')
            #ax = plt.axes(projection='3d')
            #ax.plot3D(image_label_arr_ctl_phys[0], image_label_arr_ctl_phys[1], image_label_arr_ctl_phys[2])
            #ax.plot3D(template_label_arr_ctl_phys[0], template_label_arr_ctl_phys[1], template_label_arr_ctl_phys[2])
            #plt.show()

            #Creating affine transformation matrices to apply to all images
            points_src = [[-image_label_arr_ctl_phys[0, 0], -image_label_arr_ctl_phys[1, 0], image_label_arr_ctl_phys[2, 0]], [-image_label_arr_ctl_phys[0, -1], -image_label_arr_ctl_phys[1, -1], image_label_arr_ctl_phys[2, -1]]]
            points_dest = [[-template_label_arr_ctl_phys[0, 0], -template_label_arr_ctl_phys[1, 0], template_label_arr_ctl_phys[2, 0]], [-template_label_arr_ctl_phys[0, -1], -template_label_arr_ctl_phys[1, -1], template_label_arr_ctl_phys[2, -1]]]
            
            #Input to template
            (rotation_matrix, translation_array, points_moving_reg, points_moving_barycenter) = getRigidTransformFromLandmarks(points_src, points_dest, constraints="Tx_Ty_Tz_Rz_Sz", verbose=1)
            # writing rigid transformation file
            # N.B. x and y dimensions have a negative sign to ensure compatibility between Python and ITK transfo
            affine_filename = image_label_filename.removesuffix('.nii.gz') + '_seg_label2template_affine.txt'
            text_file = open(affine_filename, 'w')
            text_file.write("#Insight Transform File V1.0\n")
            text_file.write("#Transform 0\n")
            text_file.write("Transform: AffineTransform_double_3_3\n")
            text_file.write("Parameters: %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n" % (
                rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
                rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
                rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2],
                translation_array[0, 0], translation_array[0, 1], translation_array[0, 2]))
            text_file.write("FixedParameters: %.9f %.9f %.9f\n" % (points_moving_barycenter[0],
                                                                    points_moving_barycenter[1],
                                                                    points_moving_barycenter[2]))
            text_file.close()

            #sct_register_multimodal.main(['-i', image_label_filename, '-iseg', image_label_filename, '-d', template_segmentation_path, '-dseg', template_segmentation_path, '-p', args.parameters, '-initwarp', affine_filename, '-x', 'nn'])
            
            #initialize param
            class Param:
                # The constructor
                def __init__(self):
                    self.debug = 0
                    self.outSuffix = "_reg"
                    self.padding = 5
                    self.remove_temp_files = 1

            param = Param()

            step0 = Paramreg(step='0', type='seg', algo='centermass', metric='MeanSquares', slicewise='1')
            step1 = Paramreg(step='1', type='seg', algo='centermass', metric='MeanSquares', slicewise='1')
            DEFAULT_PARAMREGMULTI = ParamregMultiStep([step0, step1])
            paramregmulti = deepcopy(DEFAULT_PARAMREGMULTI)

            args.parameters = 'step=1,type=seg,algo=centermassrot,metric=MeanSquares,slicewise=1'
            print('Errors after here')
            paramregmulti_user = args.parameters
            # update registration parameters
            for paramStep in paramregmulti_user:
                paramregmulti.addStep(paramStep)

            #sct_register_multimodal.main(['-i', image_label_filename, '-iseg', image_label_filename, '-d', template_segmentation_path, '-dseg', template_segmentation_path, '-p', args.parameters, '-initwarp', affine_filename, '-x', 'nn'])

            fname_src = image_label_filename
            fname_dest = template_segmentation_path
            fname_src_seg = image_label_filename
            fname_dest_seg = template_segmentation_path
            fname_src_label = None
            fname_dest_label = None
            fname_mask = None
            fname_initwarp = affine_filename
            fname_initwarpinv = None
            identity = None
            interp = 'nn'
            fname_output = 'output.nii.gz'
            fname_output_warp = 'warp.nii.gz'
            fname_output_warpinv = None
            path_out = None

            print('here')
            fname_src2dest, fname_dest2src, _, _ = \
            register_wrapper(fname_src, fname_dest, param, paramregmulti, fname_src_seg=fname_src_seg,
                         fname_dest_seg=fname_dest_seg, fname_src_label=fname_src_label,
                         fname_dest_label=fname_dest_label, fname_mask=fname_mask, fname_initwarp=fname_initwarp,
                         fname_initwarpinv=fname_initwarpinv, identity=identity, interp=interp,
                         fname_output=fname_output,
                         fname_output_warp=fname_output_warp, fname_output_warpinv=fname_output_warpinv,
                         path_out=path_out)


if __name__ == "__main__":
    main()