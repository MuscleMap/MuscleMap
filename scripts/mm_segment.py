#!/usr/bin/env python
# -*- coding: utf-8 -*-

# For usage, type: python mm_segment.py -h

# Authors: Richard Yin, Eddo Wesselink, Brian Kim and Kenneth Weber
# Authors: Richard Yin, Eddo Wesselink, Brian Kim and Kenneth Weber

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
from monai.utils import set_determinism
import time
from time import perf_counter
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="monai")
try:
    # Attempt to import as if it is a part of a package
    from .mm_util import check_image_exists, get_model_and_config_paths, load_model_config, validate_seg_arguments, RemapLabels,SqueezeTransform, run_inference, is_nifti, report_compute_usage, report_gpu_stats, estimate_max_chunk_slices
except ImportError:
    # Fallback to direct import if run as a standalone script
    from mm_util import check_image_exists, get_model_and_config_paths, load_model_config, validate_seg_arguments,RemapLabels,SqueezeTransform, run_inference, is_nifti, report_compute_usage, report_gpu_stats, estimate_max_chunk_slices

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
    
    optional.add_argument("-s", '--overlap', required=False, default = 50, type=float,
                        help="Percent spatial overlap during sliding window inference, higher percent may improve accuracy but will reduce inference speed. Default is 50. If accuracy needs to be improved, the spatial overlap can be increased. To improve performance, we recommend increasing the spatial inference to 90.")
    
    optional.add_argument("-c", '--chunk_size', required=False, default = 'auto', type=str,
                    help="Number of axial slices to be processed as a single chunk, or 'auto' to estimate from GPU memory. Default is 'auto'")

    return parser

# main: sets up logging, parses command-line arguments using parser, runs model, inference, post-processing
def main():
    gc.collect()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # avoid memory fragmentation
    import torch

    script_path = os.path.abspath(__file__)
    print(f"The absolute path of the script is: {script_path}")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = get_parser()
    args = parser.parse_args()
         
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_GPU=='Y' else "cpu")
    
    logging.info(f"Processing using cuda or cpu: {device}")
    
    amp_context = torch.amp.autocast('cuda') if torch.cuda.is_available() and args.use_GPU == 'Y' else nullcontext()
    
    if device.type == "cuda":
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        gpu_report = report_gpu_stats(device)
        if gpu_report:
            dev = gpu_report.get("device")
            total_mb = gpu_report.get("total_mb")
            free_mb = gpu_report.get("free_mb")
            alloc_mb = gpu_report.get("allocated_mb")
            pct_free = gpu_report.get("pct_free")
            logging.info(
                "GPU report: device=%s; total=%.0f MB; free=%.0f MB (%.1f%%); allocated=%.1f MB",
                dev,
                total_mb if total_mb is not None else 0.0,
                free_mb if free_mb is not None else 0.0,
                pct_free if pct_free is not None else 0.0,
                alloc_mb if alloc_mb is not None else 0.0,
            )
        else:
            logging.info("GPU report: unavailable")
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
        # Check that each image exists and is readable
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

    set_determinism(seed=0)  

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
        to_tensor=False, device=device
    ),
    AsDiscreted(keys="pred", argmax=True),
    SqueezeTransform(keys=["pred"])]

    test_files = [{"image": image} for image in image_paths]
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

    if args.region == 'wholebody':
        post_transforms.extend([
        RemapLabels(keys=["pred"], id_map=inv_id_map)])
    
    post_transforms = Compose(post_transforms)
  
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    overlap_inference = args.overlap / 100
    inferer = SliceInferer(roi_size=roi_size, sw_batch_size=spatial_window_batch_size, spatial_dim=2, mode="gaussian", overlap=overlap_inference)
    # args.chunk_size may be an integer string or the literal 'auto'
    chunk_size_arg = args.chunk_size
    chunk_size = None
    for test in test_files:
        logging.info(f"Processing {test['image']}")
        t0 = perf_counter()
        proc_start = time.process_time()
        # determine chunk size for this image
        if isinstance(chunk_size_arg, str) and chunk_size_arg.lower() == 'auto':
            try:
                est = estimate_max_chunk_slices(test['image'], device, pre_transforms, inferer, model, amp_context, safety_fraction=0.98)
                if est is not None and est > 0:
                    chunk_size = int(est)
                    logging.info("Auto-estimated chunk size: %d slices (98%% of free GPU VRAM)", chunk_size)
                else:
                    chunk_size = 25
                    logging.warning("Auto-estimation failed; falling back to default chunk size %d", chunk_size)
            except Exception:
                chunk_size = 25
                logging.warning("Auto-estimation raised an exception; falling back to chunk size %d", chunk_size)
        else:
            try:
                chunk_size = int(chunk_size_arg)
            except Exception:
                chunk_size = 25
        try:
            out_path = run_inference(test["image"], output_dir, pre_transforms, post_transforms, amp_context, chunk_size, device, inferer, model)
            elapsed = perf_counter() - t0
            if elapsed >= 60:
                minutes = int(elapsed // 60)
                seconds = elapsed % 60
                logging.info(f"Inference of {test} finished in {minutes}m {seconds:.2f}s")
            else:
                logging.info(f"Inference of {test} finished in {elapsed:.2f}s")    
            # capture CPU usage report and log it (Usage report)
            cpu_report = report_compute_usage(out_path, t0, proc_start, device)
            if cpu_report:
                outp = cpu_report.get("out_path")
                system_cpu = cpu_report.get("system_cpu_pct")
                proc_cpu_time = cpu_report.get("proc_cpu_time")
                process_pct = cpu_report.get("process_cpu_pct")

                # format proc_cpu_time as mm:ss when > 60s
                if proc_cpu_time is None:
                    time_str = "N/A"
                elif proc_cpu_time >= 60:
                    m = int(proc_cpu_time // 60)
                    s = proc_cpu_time % 60
                    time_str = f"{m}m {s:.2f}s"
                else:
                    time_str = f"{proc_cpu_time:.2f}s"

                logging.info(
                    "Usage report: out=%s; system_cpu=%s%%; proc_cpu_time=%s; process_cpu_pct=%s%%",
                    outp,
                    f"{system_cpu:.1f}" if system_cpu is not None else "N/A",
                    time_str,
                    f"{process_pct:.1f}" if process_pct is not None else "N/A",
                )
            else:
                logging.info("Usage report: unavailable")
        except Exception as e:
            logging.exception(f"Error processing {test['image']}: {e}")
            continue
    logging.info("Inference completed. All outputs saved.")

#%%
if __name__ == "__main__":
    main()




