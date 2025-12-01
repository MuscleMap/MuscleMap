#!/usr/bin/env python
# -*- coding: utf-8 -*-

# For usage, type: python mm_segment.py -h

# Authors: Richard Yin, Eddo Wesselink, and Kenneth Weber

# IMPORTS: necessary libraries, modules, including MONAI for image processing, argparse, and torch for Deep Learning

import argparse
import logging
import os
import gc
from contextlib import nullcontext
import sys
print("Command line arguments received:", sys.argv)
from monai.inferers import SliceInferer
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    Orientationd,
    Spacingd,
    EnsureTyped,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    CropForegroundd,
)
from monai.networks.layers import Norm
from time import perf_counter
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="monai")

try:
    # Attempt to import as if it is a part of a package
    from .mm_util import check_image_exists, get_model_and_config_paths, load_model_config, validate_seg_arguments, RemapLabels,SqueezeTransform, run_inference,is_nifti
except ImportError:
    # Fallback to direct import if run as a standalone script
    from mm_util import check_image_exists, get_model_and_config_paths, load_model_config, validate_seg_arguments,RemapLabels,SqueezeTransform, run_inference,is_nifti
import torch

#naming not functional
# get_parser: parses command line arguments, sets up a) required (image, body region), and b) optional arguments (model, output file name, output directory)
def get_parser():
    parser = argparse.ArgumentParser(
        description="Segment an input image according to the specified region.")
    
    # Required arguments
    required = parser.add_argument_group("Required")
    
    required.add_argument("-i", '--input_image', required=True, type=str,
                          help="Input image to segment. Can be single image or list of images separated by commas.")
    
    required.add_argument("-r", '--region', required=False, default = 'wholebody', type=str,
                          help="Anatomical region to segment. Supported regions: wholebody, abdomen, pelvis, thigh, and leg. Default is wholebody.")
    # Optional arguments
    optional = parser.add_argument_group("Optional")
    required.add_argument("-o", '--output_dir', required=False, type=str,
                            help="Output directory to save the results, output file name suffix = dseg. If left empty, saves to current working directory.")
    
    optional.add_argument("-m", '--model', default = None,  required=False, type=str,
                          help="Option to specify another model.")
    
    optional.add_argument("-g", '--use_GPU', required=False, default = 'Y', type=str ,choices=['Y', 'N'],
                        help="If N will use the cpu even if a cuda enabled device is identified. Default is Y.")
    
    optional.add_argument("-s", '--overlap', required=False, default = 90, type=float,
                         help="Percent spatial overlap during sliding window inference, higher percent may improve accuracy but will reduce inference speed. Default is 90. If inference speed needs to be increased, the spatial overlap can be lowered. For large high-resolution or whole-body images, we recommend lowering the spatial inference to 50.")

    optional.add_argument("-c", '--chunk_size', required=False, default = 25, type=int,
                    help="Number of axials slices to be processed as a single chunk. If image is larger than chunk size, then image will be processed in separate chunks to save memory and improve speed. Default is 50 slices.")

    return parser

# main: sets up logging, parses command-line arguments using parser, runs model, inference, post-processing
def main():
    gc.collect()
    script_path = os.path.abspath(__file__)
    print(f"The absolute path of the script is: {script_path}")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = get_parser()
    args = parser.parse_args()
         
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_GPU=='Y' else "cpu")
    
    logging.info(f"Processing using cuda or cpu: {device}")
    
    amp_context = torch.amp.autocast('cuda') if torch.cuda.is_available() and args.use_GPU == 'Y' else nullcontext()
    
    if device.type =='cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
    else:
        logging.info(f"Processing on a CPU will slow down inference speed")

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

    validate_seg_arguments(args)

    image_paths = [image.strip() for image in args.input_image.split(',')]
    for image_path in image_paths:
        logging.info(f"Checking if image '{image_path}' exists and is readable...")
        check_image_exists(image_path)
        if not is_nifti(image_path):
            logging.error(f"Error: {image_path} is not a valid NIfTI (.nii or .nii.gz)")
            sys.exit(1) 

    logging.info("Loading configuration file...")

    model_path, model_config_path = get_model_and_config_paths(args.region, args.model)

    model_config = load_model_config(model_config_path)

    norm_map = {
    "instance": Norm.INSTANCE,
    }
    try:
        roi_size = tuple(model_config['parameters']['roi_size'])
        spatial_window_batch_size = model_config['parameters']['spatial_window_batch_size']
        pix_dim = tuple(model_config['parameters']['pix_dim'])
        spatial_dims = model_config['model']['spatial_dims']
        in_channels = model_config['model']['in_channels']
        out_channels = model_config['model']['out_channels']
        channels = model_config['model']['channels']
        act = model_config['model']['act']
        strides = model_config['model']['strides']
        num_res_units = model_config['model']['num_res_units']
        import_norm_str = model_config['model']['norm']
        label_entries = model_config["labels"]
    except KeyError as e:
        logging.error(f"Missing key in model configuration file: {e}")
        sys.exit(1)
    
    labels = sorted({entry["value"] for entry in label_entries})
    id_map = {0: 0}
    for new_id, orig in enumerate(labels, start=1):
        id_map[orig] = new_id
    inv_id_map = {new_id: orig for orig, new_id in id_map.items()}

    # Directory setup
    if import_norm_str in norm_map:
        import_norm = norm_map[import_norm_str]
    else:
        logging.error(f"Unknown normalization type: {import_norm_str}")
        sys.exit(1) 

    pre_transforms = Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=pix_dim, mode="bilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        CropForegroundd(keys=["image"], source_key="image", margin=20),
        EnsureTyped(keys=["image"]),
    ])
    
    post_transforms = [
    Invertd(
        keys="pred", transform= pre_transforms, orig_keys="image",
        meta_keys="pred_meta_dict", orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict", nearest_interp=False,
        to_tensor=True, device=device
    ),
    AsDiscreted(keys="pred", argmax=True),
    SqueezeTransform(keys=["pred"])]

    test_files = [{"image": image} for image in image_paths]

    if args.region == 'wholebody':
        post_transforms.extend([
        RemapLabels(keys=["pred"], id_map=inv_id_map)])
    
    post_transforms = Compose(post_transforms)
    state = torch.load(model_path, map_location="cpu", weights_only=True)

    model = UNet(
    spatial_dims=spatial_dims,
    in_channels=in_channels,
    out_channels=out_channels,
    channels=channels,
    act=act,
    strides=strides,
    num_res_units=num_res_units,
    norm=import_norm)

    model.load_state_dict(state)
    del state
    gc.collect()
    model = model.to(device)
    model.eval()

    overlap_inference = args.overlap / 100
    inferer = SliceInferer(roi_size=roi_size, sw_batch_size=spatial_window_batch_size, spatial_dim=2, mode="gaussian", overlap=overlap_inference)
    chunk_size = args.chunk_size
    for test in test_files:
        logging.info(f"Processing {test['image']}")
        t0 = perf_counter()
        try:
            run_inference(test["image"], output_dir, pre_transforms, post_transforms, amp_context, chunk_size, device, inferer, model )
            logging.info(f"Inference of {test} finished in {perf_counter()-t0:.2f}s")
        except Exception as e:
            logging.exception(f"Error processing {test['image']}: {e}"),
            continue
# %%
    logging.info("Inference completed. All outputs saved.")
if __name__ == "__main__":
    main()

