import os
import sys
import argparse
import logging
import nibabel as nib
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import ndimage
import matplotlib

try:
    sct_dir = os.environ.get('SCT_DIR')
    sys.path.append(sct_dir)
    sct_version = np.loadtxt(os.path.join(sct_dir, 'spinalcordtoolbox', 'version.txt'), dtype='str')
    from spinalcordtoolbox.image import Image
    from spinalcordtoolbox.registration.landmarks import getRigidTransformFromLandmarks
    from spinalcordtoolbox.centerline.core import get_centerline
    from spinalcordtoolbox.resampling import resample_nib
    from spinalcordtoolbox.scripts import sct_apply_transfo, sct_concat_transfo, sct_register_multimodal
    from spinalcordtoolbox.scripts.sct_apply_transfo import Transform
    from spinalcordtoolbox.registration.algorithms import Paramreg, ParamregMultiStep
    from spinalcordtoolbox.registration.core import register_wrapper

except ImportError:
    sys.exit('Spinal Cord Toolbox not installed.')

if sct_version != '6.5':
     print('WARNING: Spinal Cord Toolbox Version 6.5 not installed.')
     print(f"WARNING: Spinal Cord Toolbox Version {sct_version} installed.")

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

def extract_centerline_points(im):
                    data_labels = im.data.astype(np.int8)
                    nb_slices = data_labels.shape[2]
                    centers = []
                    for z in range(nb_slices):
                        slice_data = data_labels[:, :, z]
                        if 1 in slice_data:
                            com = ndimage.center_of_mass(slice_data == 1)
                            centers.append([int(round(np.mean(com[0]))), int(round(com[1])), z])
                    centerlines_phys = im.transfo_pix2phys(centers)
                    return centerlines_phys

def transfo_pix2phys(im, coordi=None):
    """
    This function returns the physical coordinates of all points of 'coordi'.
    :param
    im:
    coordi: sequence of (nb_points x 3) values containing the pixel coordinate of points.
    :return: sequence with the physical coordinates of the points in the space of the image.
    Example:
    .. code:: python
        img = Image('file.nii.gz')
        coordi_pix = [[1,1,1]]   # for points: (1,1,1). N.B. Important to write [[x,y,z]] instead of [x,y,z]
        coordi_pix = [[1,1,1],[2,2,2],[4,4,4]]   # for points: (1,1,1), (2,2,2) and (4,4,4)
        coordi_phys = img.transfo_pix2phys(coordi=coordi_pix)
    """

    m_p2f = im.header.get_best_affine()
    aug = np.hstack((np.asarray(coordi), np.ones((len(coordi), 1))))
    ret = np.empty_like(coordi, dtype=np.float64)
    for idx_coord, coord in enumerate(aug):
        phys = np.matmul(m_p2f, coord)
        ret[idx_coord] = phys[:3]
    return ret

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

    template_path, template_segmentation_path = get_template_paths(args.region)

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
      
    except ImportError:
        print(f"Unable to load {args.region}_template.nii.gz")
    
    try:
        image_filename = args.input_image
        image = Image(image_filename).change_orientation("RPI")
      
    except ImportError:
        print(f"Unable to load {args.input_image}")

    try:
        image_seg_filename = args.segmentation_image
        image_seg = Image(image_seg_filename).change_orientation("RPI")
      
    except ImportError:
        print(f"Unable to load {args.segmentation_image}")
    

    outputs = {}

    labels = np.unique(image_seg.data)
        
    #labels = np.arange(1,2)
    for label in labels:

        if label > 0: #iterates through muscles 
            
            print(f"Starting registration process for label {label}")
            
            template_label_filename  = template_segmentation_path.removesuffix('.gz').removesuffix('.nii') + '_label-' + str(label) + '.nii.gz'
            template_label = Image(template_label_filename).change_orientation("RPI")

            try:

                image_label = image_seg.copy()
                image_label_mask = image_label.data == label
                image_label.data = image_label_mask * 1
                image_label_filename = os.path.join(output_dir, os.path.basename(image_filename.removesuffix('.gz').removesuffix('.nii')) + '_dseg_label-' + str(label) + '.nii.gz')
                image_label.save(image_label_filename, mutable=True, dtype='int16')
                
                print(f"\n... extracting centerline points and calculating transform")
                image_label_centerline = extract_centerline_points(image_label)
                template_label_centerline = extract_centerline_points(template_label)
                nb_im, nb_template = len(image_label_centerline), len(template_label_centerline)

                p10_im, p50_im, p90_im = int(0.1*nb_im), int(0.5*nb_im), int(0.9*nb_im)
                p10_template, p50_itemplate, p90_template = int(0.1*nb_template), int(0.5*nb_template), int(0.9*nb_template)

                points_src = [[-image_label_centerline[p10_im][0], -image_label_centerline[p10_im][1], image_label_centerline[p10_im][2]],
                            [-image_label_centerline[p50_im][0], -image_label_centerline[p50_im][1], image_label_centerline[p50_im][2]],
                            [-image_label_centerline[p90_im][0], -image_label_centerline[p90_im][1], image_label_centerline[p90_im][2]]]
                points_dest = [[-template_label_centerline[p10_template][0], -template_label_centerline[p10_template][1], template_label_centerline[p10_template][2]],
                            [-template_label_centerline[p50_itemplate][0], -template_label_centerline[p50_itemplate][1], template_label_centerline[p50_itemplate][2]],
                            [-template_label_centerline[p90_template][0], -template_label_centerline[p90_template][1], template_label_centerline[p90_template][2]]]

                #Creating affine transformation matrices to apply to all images
                # points_src = [[-image_label_arr_ctl_phys[0, 3], -image_label_arr_ctl_phys[1, 3], image_label_arr_ctl_phys[2, 3]], [-image_label_arr_ctl_phys[0, -4], -image_label_arr_ctl_phys[1, -4], image_label_arr_ctl_phys[2, -4]]]
                # points_dest = [[-template_label_arr_ctl_phys[0, 3], -template_label_arr_ctl_phys[1, 3], template_label_arr_ctl_phys[2, 3]], [-template_label_arr_ctl_phys[0, -4], -template_label_arr_ctl_phys[1, -4], template_label_arr_ctl_phys[2, -4]]]
                (rotation_matrix, translation_array, points_moving_reg, points_moving_barycenter) = getRigidTransformFromLandmarks(points_src, points_dest, constraints="Tx_Ty_Tz_Sz", verbose=1)
                # writing rigid transformation file
                # N.B. x and y dimensions have a negative sign to ensure compatibility between Python and ITK transfo
                
                affine_filename = os.path.join(output_dir, os.path.basename(image_label_filename.removesuffix('.gz').removesuffix('.nii')) + '_dseg_label-' + str(label) + '_affine.txt')

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

                print(f"\n... applying transform on image and segmentation")
                # Apply Affine transform to the original image to bring it to the template
                # crop= 0: no reference, 1: sets background to 0, 2: use normal background.
                # interp= 'nn', 'linear', 'spline', 'label'
                # run_proc()             

                image_affine_filename = os.path.join(output_dir, os.path.basename(image_filename.removesuffix('.gz').removesuffix('.nii')) + '_label-' + str(label) + '_affine' + '.nii.gz')

                transform = Transform(input_filename=image_filename, fname_dest=template_path, list_warp=[affine_filename],
                                    output_filename=image_affine_filename, crop=0, interp='linear', verbose=0)
                transform.apply()

                image_label_affine_filename = os.path.join(output_dir, os.path.basename(image_filename.removesuffix('.gz').removesuffix('.nii')) + '_dseg_label-' + str(label) + '_affine' + '.nii.gz')

                transform = Transform(input_filename=image_label_filename, fname_dest=template_path, list_warp=[affine_filename],
                                    output_filename=image_label_affine_filename, crop=0, interp='nn')
                transform.apply()
                
                print(f"Check affine transformation if needed: fsleyes {template_path} {image_affine_filename} {template_label_filename} {image_label_affine_filename} &")

                print(f"\n... registration of the image and the template")
                
                sct_register_multimodal.main(['-i', image_affine_filename,
                                            '-iseg', image_label_affine_filename,
                                            '-d', template_path,
                                            '-dseg', template_label_filename,
                                            '-param',
                                            'step=1,type=seg,algo=centermass,metric=MeanSquares:'
                                            'step=2,type=seg,algo=bsplinesyn,metric=MeanSquares,iter=5,shrink=2,smooth=1,gradStep=1:'
                                            'step=3,type=seg,algo=bsplinesyn,metric=MeanSquares,iter=5',
                                            '-ofolder', output_dir])

                print(f"\n... applying transformation field to segmentation")

                print(f"\n image_affine_filename {image_affine_filename}")
                print(f"\n template_path {template_path}")
                print(f"\n image_label_filename {image_label_filename}")

                
                image_label_affine_filename = os.path.join(output_dir, os.path.basename(image_filename.removesuffix('.gz').removesuffix('.nii')) + '_dseg_label-' + str(label) + '_affine' + '.nii.gz')
                warp_image_affine2template_filename = os.path.join(output_dir, 'warp_' + os.path.basename(image_filename.removesuffix('.gz').removesuffix('.nii')) + '_label-' + str(label) + '_affine2' + args.region + '_template.nii.gz')   
                image_label_affine2template_filename = os.path.join(output_dir, os.path.basename(image_filename.removesuffix('.gz').removesuffix('.nii')) + '_dseg_label-' + str(label) + '_affine2' + args.region + '_template.nii.gz')

                sct_apply_transfo.main(['-i', image_label_affine_filename,
                                        '-d', template_path,
                                        '-w', warp_image_affine2template_filename,
                                        '-x', 'linear',
                                        '-o', image_label_affine2template_filename])

                print(f"\n... concatenating transformation fields from image to template")
                
                warp_image2template_filename = os.path.join(output_dir, 'warp_' + os.path.basename(image_filename.removesuffix('.gz').removesuffix('.nii')) + '_label-' + str(label) + '2' + args.region + '_template.nii.gz')
                
                sct_concat_transfo.main(['-d', template_path,
                                        '-w', affine_filename, warp_image_affine2template_filename,
                                        '-o', warp_image2template_filename])

                print(f"\n... applying transformation field to original image")
                image2template_filename = os.path.join(output_dir, os.path.basename(image_filename.removesuffix('.gz').removesuffix('.nii')) + '_label-' + str(label) + '2' + args.region + '_template.nii.gz')
                sct_apply_transfo.main(['-i', image_filename,
                                        '-d', template_path,
                                        '-w', warp_image2template_filename,
                                        '-x', 'linear',
                                        '-o', image2template_filename])
                
                print(f"\n... applying transformation field to original image label")

                image_label2template_filename = os.path.join(output_dir, os.path.basename(image_filename.removesuffix('.gz').removesuffix('.nii')) + '_dseg_label-' + str(label) + '2' + args.region + '_template.nii.gz')
                                
                sct_apply_transfo.main(['-i', image_label_filename,
                                        '-d', template_path,
                                        '-w', warp_image2template_filename,
                                        '-x', 'nn',
                                        '-o', image_label2template_filename])

                os.rename(os.path.join(output_dir, args.region + '_template_reg.nii.gz'), os.path.join(output_dir, os.path.basename(args.region + '_template2' + image_filename.removesuffix('.gz').removesuffix('.nii')) + '_label-' + str(label) + '.nii.gz'))

                # outputs[label] = {'warp_im2template': warp_image2template_filename,
                #                 'fname_image_in_template': image2template_filename,
                #                 'image_label2template_filename': image_label2template_filename}

                print(f"... finished registration for label {label}")

            except RuntimeError:
                 
                print(f"Error registering label {label}")

                continue

    # list all outputs
    # print("\nAll generated files:")
    # for label in labels:
    #     print("Label 1", outputs[label])

if __name__ == "__main__":
    main()
