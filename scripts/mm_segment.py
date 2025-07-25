 #!/usr/bin/env python
# -*- coding: utf-8 -*-

# For usage, type: python mm_segment.py -h

# Authors: Richard Yin, Eddo Wesselink, and Kenneth Weber

# IMPORTS: necessary libraries, modules, including MONAI for image processing, argparse, and torch for Deep Learning

import argparse
import logging
import os
import gc
import sys
import numpy as np
print("Command line arguments received:", sys.argv)
import pandas as pd
import glob
import shutil
import nibabel as nib
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
    SaveImaged,
    KeepLargestConnectedComponentd,
)
from monai.networks.layers import Norm
from monai.utils import set_determinism
from monai.data import Dataset, decollate_batch, ThreadDataLoader
try:
    # Attempt to import as if it is a part of a package
    from .mm_util import check_image_exists, get_model_and_config_paths, load_model_config, validate_seg_arguments, create_output_dir, RemapLabels,SqueezeTransform
except ImportError:
    # Fallback to direct import if run as a standalone script
    from mm_util import check_image_exists, get_model_and_config_paths, load_model_config, validate_seg_arguments, create_output_dir,RemapLabels,SqueezeTransform
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
    
    required.add_argument("-r", '--region', required=True, type=str,
                          help="Anatomical region to segment. Supported regions: abdomen, pelvis, thigh, wholebody and leg")
    
    # Optional arguments
    optional = parser.add_argument_group("Optional")
    required.add_argument("-o", '--output_dir', required=False, type=str,
                            help="Output directory to save the results, output file name suffix = dseg. If left empty, saves to current working directory.")
    
    optional.add_argument("-m", '--model', default = None,  required=False, type=str,
                          help="Option to specify another model.")
    
    optional.add_argument("-g", '--use_GPU', required=False, default = 'Y', type=str ,choices=['Y', 'N'],
                        help="If N will use the cpu even if a cuda enabled device is identified. Default is Y.")
    
    optional.add_argument("-c", '--chunk_size', required=False, default = 50, type=int,
                    help="Determine the chunk size, if larger than chunk-size, then model will process in seperate chunks to save memory and improve speed")

    return parser

# main: sets up logging, parses command-line arguments using parser, runs model, inference, post-processing
def main():
    script_path = os.path.abspath(__file__)
    print(f"The absolute path of the script is: {script_path}")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = get_parser()
    args = parser.parse_args()
    logging.info(f"I'm using a GPU (yes/no): {args.use_GPU}")

    if args.use_GPU == 'Y':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = True

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

    # Process multiple images
    image_paths = [image.strip() for image in args.input_image.split(',')]
    for image_path in image_paths:
        # Check that each image exists and is readable
        logging.info(f"Checking if image '{image_path}' exists and is readable...")
        check_image_exists(image_path)
    
        ####here

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
    }

    try:
        roi_size = tuple(model_config['parameters']['roi_size'])
        spatial_window_batch_size = model_config['parameters']['spatial_window_batch_size']
        pix_dim = tuple(model_config['parameters']['pix_dim'])

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

    set_determinism(seed=0)  # Seed for reproducibility (identical to training part)
    
    inference_transforms = Compose([
        LoadImaged(keys=["image"], image_only = False),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=pix_dim,
            mode=("bilinear")),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        EnsureTyped(keys=["image"])
    ])

    # Process all images at once
    test_files = [{"image": image} for image in image_paths]

    # Device config
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_GPU=='Y' else "cpu")

    if args.region != 'wholebody':
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
            KeepLargestConnectedComponentd(keys="pred", applied_labels=list(range(1, num_labels + 1))),
            SaveImaged(
                keys="pred", 
                meta_keys="pred_meta_dict", 
                output_dir=output_dir, 
                output_dtype=('int16'), 
                separate_folder=False, 
                resample=False, 
                output_postfix="dseg",
        )
        ])
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
    else: 
        logging.info("I'm loading in the current labels and model of the wholebody model")
        script_dir = os.path.dirname(__file__)
        excel_path = os.path.join(script_dir, "models", "wholebody", "MuscleMap_labels.xlsx")
        sheets = ["Leg", "Thigh", "Hip", "Lumbar", "Cervical", "Shoulder", "Thorax"]
        labels = []
        xl = pd.ExcelFile(excel_path)
        for sh in sheets:
            df = xl.parse(sh)
            labels.extend(df["MuscleMap_label"].astype(int).tolist())
        labels = sorted(set(labels))
        id_map = {0: 0}
        for new_id, orig in enumerate(labels, start=1):
            id_map[orig] = new_id
        inv_id_map = {new_id: orig_id for orig_id, new_id in id_map.items()}

        # setting the posttransforms including remap labels and additional squeezetransform
        post_transforms = Compose([
        Invertd(
            keys="pred", transform= inference_transforms, orig_keys="image",
            meta_keys="pred_meta_dict", orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict", nearest_interp=False,
            to_tensor=False, device="cuda"
        ),
        AsDiscreted(keys="pred", argmax=True),
        KeepLargestConnectedComponentd(keys="pred", applied_labels=list(range(1, len(labels)+1))),
        RemapLabels(keys=["pred"], id_map=inv_id_map),
        SqueezeTransform(keys=["pred"])])

        # Create model and set parameters
        model = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=len(labels)+1,
            channels=channels,
            act=act,
            strides=strides,
            num_res_units=num_res_units,
            norm=import_norm,
        ).to(device)
        model = model.half()

    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=map_location, weights_only=True))
    model.eval()
    inferer = SliceInferer(roi_size=roi_size, sw_batch_size=spatial_window_batch_size, spatial_dim=2, mode="gaussian", overlap=0.90)

  # Process each image
    chunk_size = args.chunk_size
    for i, test_file in enumerate(test_files):
        logging.info(f' I am processing {test_file}')
        try:
            # Load image to check slice count
            img_nii = nib.load(test_file["image"])
            img_data = img_nii.get_fdata()
            num_slices = img_data.shape[-1]
            if num_slices <= chunk_size:
                logging.info('Im processing the entire image')
                dataset = Dataset(data=[test_file], transform=inference_transforms)
                loader = ThreadDataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False, num_workers=0)
                with torch.no_grad():
                    for test_data in loader:
                        if 'image_meta_dict' not in test_data:
                            test_data['image_meta_dict'] = {'filename_or_obj': 'unknown'}
                        with torch.amp.autocast('cuda'):
                            val_inputs = test_data["image"].half().to(device, non_blocking=True)
                            test_data["pred"] = inferer(val_inputs, model)
                            raw_pred = torch.argmax(test_data["pred"], dim=1).cpu().numpy()
                            val_data = [post_transforms(i) for i in decollate_batch(test_data)]
                            pred_data = val_data[0]["pred"].cpu().numpy().astype(np.int16)
                            output_file = os.path.join(output_dir, os.path.basename(test_file["image"]).replace(".nii.gz", "_dseg.nii.gz"))
                            nib.save(nib.Nifti1Image(pred_data, img_nii.affine, img_nii.header), output_file)
                        del val_inputs, test_data, val_data, raw_pred, pred_data
                        gc.collect()
                        torch.cuda.empty_cache()
            else:
                logging.info('I will split and process the image per chunk to improve efficiency')
                temp_chunk_dir = os.path.join(output_dir, 'temp_chunks')
                os.makedirs(temp_chunk_dir, exist_ok=True)
                chunk_files = []
                for start in range(0, num_slices, chunk_size):
                    end = min(start + chunk_size, num_slices)
                    chunk_file = os.path.join(temp_chunk_dir, f"chunk_{start}_{end}.nii.gz")
                    chunk_data = img_data[..., start:end]
                    chunk_nii = nib.Nifti1Image(chunk_data, img_nii.affine, img_nii.header)
                    nib.save(chunk_nii, chunk_file)
                    chunk_files.append({"image": chunk_file, "start": start, "end": end})
                dataset = Dataset(data=chunk_files, transform=inference_transforms)
                loader = ThreadDataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False, num_workers=0)
                chunk_seg_data = []
                with torch.no_grad():
                    for test_data in loader:
                        start = test_data["start"][0].item()  
                        end = test_data["end"][0].item()      
                        with torch.amp.autocast('cuda'):
                            val_inputs = test_data["image"].half().to(device, non_blocking=True)
                            test_data["pred"] = inferer(val_inputs, model)
                            raw_pred = torch.argmax(test_data["pred"], dim=1).cpu().numpy()
                            val_data = [post_transforms(i) for i in decollate_batch(test_data)]
                            # Save chunk segmentation to disk
                            chunk_pred_data = val_data[0]["pred"].cpu().numpy().astype(np.int16)
                            # Store in memory for merging
                            chunk_seg_data.append(chunk_pred_data)
                        del val_inputs, test_data, val_data, raw_pred, chunk_pred_data
                        gc.collect()
                        torch.cuda.empty_cache()
                # Merge chunk segmentations
                merged_data = np.concatenate(chunk_seg_data, axis=-1)
                output_file = os.path.join(output_dir, os.path.basename(test_file["image"]).replace(".nii.gz", "_dseg.nii.gz"))
                nib.save(nib.Nifti1Image(merged_data, img_nii.affine, img_nii.header), output_file)
                for chunk_file in chunk_files:
                    try:
                        os.remove(chunk_file["image"])
                    except Exception as e:
                        logging.warning(f"Could not remove temporary image file {chunk_file['image']}: {e}")
                if os.path.exists(temp_chunk_dir):
                    os.rmdir(temp_chunk_dir)
        except Exception:
            logging.exception(f"Error processing {image_path}, skipping.")
            continue

    logging.info("Inference completed. All outputs saved.")
if __name__ == "__main__":
    main()
