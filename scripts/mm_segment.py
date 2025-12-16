#!/usr/bin/env python
# -*- coding: utf-8 -*-

# For usage, type: python mm_segment.py -h

# Authors: Richard Yin, Eddo Wesselink, Brian Kim and Kenneth Weber

import warnings
import argparse
warnings.filterwarnings("ignore")
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
try:
    # Attempt to import as if it is a part of a package
    from .mm_util import check_image_exists, get_model_and_config_paths, load_model_config, validate_seg_arguments, RemapLabels,SqueezeTransform, run_inference, run_inference_fast, is_nifti, estimate_chunk_size
except ImportError:
    # Fallback to direct import if run as a standalone script
    from mm_util import check_image_exists, get_model_and_config_paths, load_model_config, validate_seg_arguments,RemapLabels,SqueezeTransform, run_inference, run_inference_fast, is_nifti, estimate_chunk_size
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
    
    optional.add_argument("-s", '--overlap', required=False, default = 75, type=float,
                         help="Percent spatial overlap during sliding window inference, higher percent may improve accuracy but will reduce inference speed. Default is 75. If inference speed needs to be increased, the spatial overlap can be lowered. For large high-resolution or whole-body images, we recommend lowering the spatial inference to 50.")
    
    optional.add_argument("-c", '--chunk_size', required=False, default = 25, type=str,
                    help="Number of axial slices to be processed as a single chunk, or 'auto' to estimate from CPU or GPU memory. Default is 25")
    
    optional.add_argument("--fast", action='store_true',
                    help="Enable fast mode: reduces overlap to 50%% and uses optimized inference without verbose tracking. Prioritizes speed over accuracy.")
    
    optional.add_argument("--verbose", action='store_true',
                    help="Enable verbose output during inference.")

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
        to_tensor=False, device="cpu"  # Use CPU to avoid OOM during inverse transforms
    ),
    AsDiscreted(keys="pred", argmax=True),
    SqueezeTransform(keys=["pred"])]

    test_files = [{"image": image} for image in image_paths]

    if args.region == 'wholebody':
        post_transforms.extend([
            RemapLabels(keys=["pred"], id_map=inv_id_map)])
    
    post_transforms = Compose(post_transforms)

    model = UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        norm=import_norm,
        act=act
    ).to(device)
  
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Convert model to FP16 which CUDA tensor cores are optimized for
    if device.type == 'cuda':
        model = model.half()
    
    # Compile model for CPU optimization (PyTorch 2.0+)
    # Note: torch.compile only supports CPU and CUDA, not MPS
    if device.type == 'cpu':
        try:
            logging.info("Compiling model with torch.compile for CPU optimization...")
            model = torch.compile(model, mode='max-autotune')
            logging.info("Model compilation successful")
        except Exception as e:
            logging.warning(f"torch.compile not available or failed: {e}. Continuing without compilation.")

    # Apply fast mode settings
    # Default overlap is 75%, but --fast mode uses 50% unless user specifies otherwise
    if args.fast and args.overlap != 75:
        # User specified both --fast and custom --overlap (e.g., --fast -s 60), use custom value
        overlap_inference = args.overlap / 100
        logging.info(f"Fast mode enabled with custom overlap: {args.overlap}%")
    elif args.fast:
        # Fast mode with default overlap (user didn't specify -s), use 50%
        overlap_inference = 0.50
        logging.info("Fast mode enabled: using 50% overlap")
    else:
        # Normal mode, use specified overlap (default 75%)
        overlap_inference = args.overlap / 100
    
    # Create SliceInferer (2D model)
    inferer = SliceInferer(roi_size=roi_size, sw_batch_size=spatial_window_batch_size, spatial_dim=2, mode="gaussian", overlap=overlap_inference)
    
    chunk_size_arg = args.chunk_size # args.chunk_size may be an integer string or the literal 'auto'
    
    errors_occurred = False
    for test in test_files:
        logging.info(f"Processing {test['image']}")
        t0 = perf_counter()
        
        # Use optimized disk-based chunking with auto chunk size
        if isinstance(chunk_size_arg, str) and chunk_size_arg.lower() == 'auto':
            # Load image header to get shape for OOM history lookup
            import nibabel as nib
            img_nii = nib.load(test['image'])
            img_shape = img_nii.header.get_data_shape()
            del img_nii  # Free memory
            
            use_fp16 = (device.type == 'cuda' and amp_context.fast_dtype == torch.float16)
            chunk_size = estimate_chunk_size(
                device=device,
                roi_size=roi_size,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=channels,
                spatial_window_batch_size=spatial_window_batch_size,
                overlap=overlap_inference,
                use_fp16=use_fp16,
                image_shape=img_shape
            )
        else:
            chunk_size = int(chunk_size_arg)
        try:
            # Create CPU fallback device for automatic OOM handling (for both CUDA and MPS)
            fallback_device = torch.device('cpu') if device.type in ['cuda', 'mps'] else None
            
            # Use fast inference mode if --fast flag is set
            if args.fast:
                run_inference_fast(
                    test["image"], 
                    output_dir, 
                    pre_transforms, 
                    post_transforms, 
                    amp_context, 
                    chunk_size, 
                    device, 
                    inferer, 
                    model, 
                    fallback_device=fallback_device
                )
                method = "fast mode"
            else:
                run_inference(
                    test["image"], 
                    output_dir, 
                    pre_transforms, 
                    post_transforms, 
                    amp_context, 
                    chunk_size, 
                    device, 
                    inferer, 
                    model, 
                    verbose=args.verbose,
                    fallback_device=fallback_device
                )
                method = "disk-based chunking" if chunk_size else "full volume"
            
            inference_time = perf_counter()-t0
            logging.info(f"Inference of {test['image']} finished in {inference_time:.2f}s ({method})")

        except Exception as e:
            logging.exception(f"Error processing {test['image']}: {e}")
            errors_occurred = True
            
    if not errors_occurred:
        logging.info("Inference completed. All outputs saved.")        

#%%
if __name__ == "__main__":
    main()