#!/usr/bin/env python

import sys
import os
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.scripts.sct_apply_transfo import main as sct_apply_transfo
from spinalcordtoolbox.scripts.sct_register_multimodal import main as sct_register_multimodal

def get_parser():
    parser = argparse.ArgumentParser(description="Register muscle segmentation image to template")

    parser.add_argument("-i", '--input_image', required=True, type=str, 
                        help="Input image in NIfTI format (e.g., image.nii.gz)")
    parser.add_argument("-s", '--segmentation_image', required=True, type=str, 
                        help="Segmentation image in NIfTI format (e.g., segmentation_image.nii.gz)")
    parser.add_argument("-r", '--region', required=True, type=str, 
                        help="Region name for template registration")
    parser.add_argument("-t", '--template_dir', required=True, type=str, 
                        help="Directory containing the template images")
    parser.add_argument("-o", '--output_dir', required=True, type=str, 
                        help="Output directory to save the results")

    return parser

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = get_parser()
    arguments = parser.parse_args(args)

    fname_image = arguments.input_image
    fname_segmentation = arguments.segmentation_image
    region = arguments.region
    template_dir = arguments.template_dir
    output_dir = arguments.output_dir

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Register segmentation image to template
    sct_register_multimodal([
        '-i', fname_segmentation,
        '-d', os.path.join(template_dir, f'{region}_template.nii.gz'),
        '-iseg', os.path.join(template_dir, f'{region}_segmentation.nii.gz'),
        '-o', os.path.join(output_dir, f'warp_template2{region}.nii.gz'),
        '-owarp', os.path.join(output_dir, f'warp_{region}2template.nii.gz')
    ])

    # Warp input image to template
    sct_apply_transfo([
        '-i', fname_image,
        '-d', os.path.join(template_dir, f'{region}_template.nii.gz'),
        '-w', os.path.join(output_dir, f'warp_{region}2template.nii.gz'),
        '-o', os.path.join(output_dir, f'image_{region}_warped.nii.gz')
    ])

    # Warp segmentation image to template
    sct_apply_transfo([
        '-i', fname_segmentation,
        '-d', os.path.join(template_dir, f'{region}_template.nii.gz'),
        '-w', os.path.join(output_dir, f'warp_{region}2template.nii.gz'),
        '-o', os.path.join(output_dir, f'segmentation_{region}_warped.nii.gz')
    ])

if __name__ == "__main__":
    main()
