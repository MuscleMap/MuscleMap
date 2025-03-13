 #!/usr/bin/env python
# -*- coding: utf-8 -*-

# For usage, type: python mm_segment.py -h

# Authors: Richard Yin, Eddo Wesselink, and Kenneth Weber

# IMPORTS: necessary libraries, modules, including MONAI for image processing, argparse, and torch for Deep Learning
import argparse
import logging
import os
import sys
print("Command line arguments received:", sys.argv)

import glob
import shutil
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
    FillHolesd,
    SaveImaged,
    KeepLargestConnectedComponentd,
)
from monai.networks.layers import Norm
from monai.utils import set_determinism
from monai.data import Dataset, DataLoader, decollate_batch

import numpy as np
import torch
import pickle
from tqdm import tqdm

try:
    # Attempt to import as if it is a part of a package
    from .mm_util import check_image_exists, get_model_and_config_paths, load_model_config, validate_seg_arguments, create_output_dir
    from .mm_reorient import get_reroientantion_order, reorient_to_in_plane_high_res
except ImportError:
    # Fallback to direct import if run as a standalone script
    from mm_util import check_image_exists, get_model_and_config_paths, load_model_config, validate_seg_arguments, create_output_dir
    from mm_reorient import get_reorientation_order, reorient_to_in_plane_high_res
import torch

#naming not functional
# get_parser: parses command line arguments, sets up a) required (image, body region), and b) optional arguments (model, output file name, output directory)
def get_parser():
    parser = argparse.ArgumentParser(
        description="Segment an input image according to the specified region.")
    
    # Required arguments
    required = parser.add_argument_group("Required")
    # Changed inputs to also accept a directory of images
    required.add_argument("-i", '--input_image', required=True, type=str,
                          help="Input image to segment. Can be single image, directory of images, or a list of images separated by commas.")
    
    required.add_argument("-r", '--region', required=True, type=str,
                          help="Anatomical region to segment. Supported regions: abdomen, pelvis, thigh, and leg")
    
    # Optional arguments
    optional = parser.add_argument_group("Optional")
    required.add_argument("-o", '--output_dir', required=False, type=str,
                            help="Output directory to save the results, output file name suffix = dseg. If left empty, saves to current working directory.")
    
    optional.add_argument("-m", '--model', default=None, required=False, type=str,
                          help="Option to specify another model.")
    
    optional.add_argument("-g", '--use_GPU', required=False, default = 'Y', type=str ,choices=['Y', 'N'],
                          help="If N will use the cpu even if a cuda enabled device is identified. Default is Y.")
    # Allows for models and inputs with swapped axes to be used
    optional.add_argument("-v", "--variable_axis", required=False, default='N', type=str, 
                          help="If True, will use the variable axis inference method. Default is False.")
    return parser

# main: sets up logging, parses command-line arguments using parser, runs model, inference, post-processing
def main():
    script_path = os.path.abspath(__file__)
    print(f"The absolute path of the script is: {script_path}")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = get_parser()
    args = parser.parse_args()
    print(f"GPU arg: {args.use_GPU}")

    base_name = os.path.basename(args.input_image)

    # Split the name from its extension
    name, ext = os.path.splitext(base_name)
    
    # Further split in case of double extensions like .nii.gz
    if ext == ".gz" and name.endswith(".nii"):
        name, _ = os.path.splitext(name)  # Remove .nii part

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

    # Validate Arguments
    validate_seg_arguments(args)

    # Process multiple images (can be a directory or list)
    input_path = args.input_image

    if os.path.isdir(input_path):
        image_paths = glob.glob(os.path.join(input_path, '*.nii.gz'))
        image_paths = [path for path in image_paths if not os.path.basename(path).startswith('.')]
    else:
        image_paths = [image.strip() for image in input_path.split(',')]
    
    for image_path in image_paths:
        logging.info(f"Checking if image '{os.path.basename(image_path)}' exists and is readable...")
        check_image_exists(image_path)

    # If variable_axis is 'Y', reorient images and labels to in-plane high-resolution axes. Use these files for inference.
    if args.variable_axis == 'Y':
        print("Variable axis is set to 'Y'. Reorienting images...")
        reoriented_output_dir = os.path.join(output_dir, 'data_reoriented')
        os.makedirs(reoriented_output_dir, exist_ok=True)

        for image_path in tqdm(image_paths, desc='Reorienting images to in-plane high-resolution axes'):
            image_output_path = os.path.join(reoriented_output_dir, os.path.relpath(image_path, input_path))
            os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
            reorient_to_in_plane_high_res(image_path, image_output_path)

        # Update image_paths to point to reoriented images
        image_paths = glob.glob(os.path.join(reoriented_output_dir, '*.nii.gz'))


    # Load model configuration
    logging.info("Loading configuration file...")

    # Get model and config paths
    model_path, model_config_path = get_model_and_config_paths(args.region, args.model)

    # Load model configuration
    model_config = load_model_config(model_config_path)

    # maps norm from json for use in model because monai imports can't be saved in json
    norm_map = {
    "batch": Norm.BATCH,
    "instance": Norm.INSTANCE,
    # Add other normalization types if needed
    }

    #Custom Name Splitter Not Used
    # def custom_name_formatter(metadata, saver, output_file_path):
    #     base_dir, base_file = os.path.split(output_file_path)
    #     file_name, ext = os.path.splitext(base_file)
    #     return {
    #         'subject': file_name

    #     }

    try:
        roi_size = tuple(model_config['parameters']['roi_size'])
        spatial_window_batch_size = model_config['parameters']['spatial_window_batch_size']
        amount_of_labels = model_config['parameters']['amount_of_labels'] #not used
        pix_dim = tuple(model_config['parameters']['pix_dim'])
        model_continue_training = model_config['parameters']['model_continue_training'] #not used

        # Load model configuration parameters
        spatial_dims = model_config['model']['spatial_dims']
        in_channels = model_config['model']['in_channels']
        out_channels = model_config['model']['out_channels']
        channels = model_config['model']['channels']
        act = model_config['model']['act']
        strides = model_config['model']['strides']
        num_res_units = model_config['model']['num_res_units']
        num_labels = model_config['model']['num_labels']
        import_norm_str = model_config['model']['norm']
    except KeyError as e:
        logging.error(f"Missing key in model configuration file: {e}")
        sys.exit(1)

    # Directory setup

    if import_norm_str in norm_map:
        import_norm = norm_map[import_norm_str]
    else:
        logging.error(f"Unknown normalization type: {import_norm_str}")
        sys.exit(1) 

    # Use os.path.join for all path constructions to ensure cross-platform compatibility
    # Set seed for reproducibility (identical to training part)
    set_determinism(seed=0)  # Seed for reproducibility (identical to training part)

    # Create transforms identical to training part, but here we don't specify the label key
    inference_transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(
            keys=["image"],
            pixdim=pix_dim,
            mode=("bilinear")),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        EnsureTyped(keys=["image"])
    ]

    # Only reorients if needed, inserting transform at same spot
    if args.variable_axis == 'N':
        inference_transforms.insert(3, Orientationd(keys=["image"], axcodes="RAS"))

    inference_transforms = Compose(inference_transforms)

    # Process all images at once
    test_files = [{"image": image} for image in image_paths]

    # Create iterable dataset and dataloader, identical to training part
    inference_transforms_dataset = Dataset(
        data=test_files, transform=inference_transforms,
    )

    inference_transforms_loader = DataLoader(
        inference_transforms_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


    # Device config
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_GPU=='Y' else "cpu")


    # Ensure flexibility to handle models with varying label counts
    # Validate post-processing steps based on specific requirements of different models
    post_transforms = Compose([
        Invertd(
            keys="pred",
            transform=inference_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
            device=device
        ),
        AsDiscreted(keys="pred", argmax=True),
        FillHolesd(keys="pred", applied_labels=list(range(1, num_labels + 1))),  # dynamic num_labels
        KeepLargestConnectedComponentd(keys="pred", applied_labels=list(range(1, num_labels + 1))),
        SaveImaged(
            keys="pred", 
            meta_keys="pred_meta_dict", 
            output_dir=output_dir, 
            output_dtype=('int16'), 
            separate_folder=False, 
            resample=False, 
            output_postfix="dseg",
            #output_name_formatter=lambda meta, saver: custom_name_formatter(meta, saver, output_file_name)

    )
    ])

    # Create model and set parameters
    model = UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        act=act,
        strides=strides,
        num_res_units=num_res_units,

        norm=import_norm,
    ).to(device)

    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Allowlist the required globals
    torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype, np.ndarray, np.float64])
    
    # Load the model weights with weights_only=True
    try:
        checkpoint = torch.load(model_path, map_location=map_location, weights_only=False)
    except pickle.UnpicklingError as e:
        logging.error(f"Failed to load weights only: {e}")
        sys.exit(1)
    
    if args.variable_axis=='Y': # The model checkpoint in MONAI training is not nested in a state_dict
        state_dict = checkpoint
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
        missing_keys, _ = model.load_state_dict(filtered_state_dict, strict=False)
    else:
        # Extract only the state_dict part of the checkpoint
        state_dict_key_name = 'state_dict'
        if 'state_dict' not in checkpoint:
            state_dict_key_name = 'network_weights'
        state_dict = checkpoint[state_dict_key_name]
        
        # Filter out unnecessary keys from state_dict that do not match with the current model's state_dict keys.
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
        
        # Load filtered state dict into current UNet instance.
        missing_keys, _ = model.load_state_dict(filtered_state_dict, strict=False)

    model.eval()




    # Run inference on all images using model, post-process the predictions
    with torch.no_grad():
        for i, input_data in enumerate(inference_transforms_loader):
            if 'image_meta_dict' not in input_data:
                input_data['image_meta_dict'] = {'filename_or_obj': 'unknown'}
            val_inputs = input_data["image"].to(device)
            axial_inferer = SliceInferer(roi_size=roi_size, sw_batch_size=spatial_window_batch_size, spatial_dim=2)
            input_data["pred"] = axial_inferer(val_inputs, model)
            val_data = [post_transforms(i) for i in decollate_batch(input_data)]

    logging.info("Inference completed. All outputs saved.")

if __name__ == "__main__":
    main()
