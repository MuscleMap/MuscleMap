import os
import pandas as pd
import argparse
import numpy as np
import logging
import math
try:
    # Attempt to import as if it is a part of a package
    from .mm_util import (get_model_and_config_paths, load_model_config, validate_extract_args, 
    extract_image_data, create_output_dir,  calculate_metrics_dixon, 
    calculate_metrics_average,calculate_metrics_thresholding,build_entry_dict_metrics,results_entry_to_dataframe,
    add_slice_counts)
except ImportError:
    # Fallback to direct import if run as a standalone script
    from mm_util import (get_model_and_config_paths, load_model_config, validate_extract_args, 
                         extract_image_data, create_output_dir,
                         calculate_metrics_dixon, calculate_metrics_average,
                         calculate_metrics_thresholding, build_entry_dict_metrics, results_entry_to_dataframe,
                         add_slice_counts)
def get_parser():
    parser = argparse.ArgumentParser(description="Extract metrics of muscle size and composition")

    parser.add_argument("-m", '--method', required=True, type=str, choices=['dixon', 'kmeans', 'gmm', 'average'], 
                          help="Method to use: kmeans, gmm, dixon, or average")

    parser.add_argument("-i", '--input_image', required=False, type=str, 
                          help="Input image for kmeans, gmm, or average metohd")
    
    parser.add_argument("-f", '--fat_image', required=False, type=str, 
                          help="Fat image for Dixon method")

    parser.add_argument("-w", '--water_image', required=False, type=str, 
                          help="Water image for Dixon method")
    
    parser.add_argument("-s", '--segmentation_image', required=False, type=str, 
                          help="Segmentation image for any method")
    
    parser.add_argument("-c", '--components', required=False, default=None, type=int, choices=[2, 3], 
                          help="Number of components for kmeans or gmm (2 or 3)")
    
    parser.add_argument("-r", '--region', required=False, type=str,
                          help="Anatomical region. Supported regions: wholebody, abdomen, pelvis, thigh, and leg")

    parser.add_argument("-o", '--output_dir', required=False, type=str, 
                          help="Output directory to save the results")

    return parser

def main():
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    args = parser.parse_args()
    validate_extract_args(args)

    logging.info(f"Method selected: {args.method}")
    logging.info(f"Segmentation image: {args.segmentation_image}")

    _, mask, affine,header, image_dim, pix_dim = extract_image_data(args.segmentation_image)

    output_dir=create_output_dir(args.output_dir)

    if args.region:
        _, model_config_path = get_model_and_config_paths(args.region, None)
        model_config = load_model_config(model_config_path)
    else:
        model_config =  None
    
    if args.method == 'dixon':
        input_filename = os.path.basename(args.fat_image)
    else:
        input_filename = os.path.basename(args.input_image)

    id_part = input_filename[:-7] if input_filename.endswith('.nii.gz') else input_filename 
    
    results_entry = build_entry_dict_metrics(
    label_img=mask,      
    model_config=model_config,
    region=True)

    # calculate number of slices with segmentation and update results_entry dictionary.
    results_entry = add_slice_counts(results_entry, mask, pix_dim)

    if not np.any(mask):
        raise ValueError("No labels found in segmentation mask")
    else:
        if args.method == 'dixon': 
            _, fat_array, _, _,_,_  = extract_image_data(args.fat_image)
            _, water_array, _, _,_,_ = extract_image_data(args.water_image)
            outputs = calculate_metrics_dixon(results_entry, mask, fat_array, water_array, pix_dim)
        elif args.method == 'average':
            _, image_array,_,_, _, _ = extract_image_data(args.input_image)
            outputs = calculate_metrics_average(results_entry, mask, image_array, pix_dim)
        elif args.method in ('kmeans', 'gmm'):
            _, image_array, _, _, _, _, = extract_image_data(args.input_image) 
            number_of_components = args.components
            outputs = calculate_metrics_thresholding(args, results_entry, mask, image_array, affine,header, 
                                                               pix_dim, number_of_components, output_dir, 
                                                                id_part)
    # Construct the path to the output CSV file
    if args.method != 'dixon' and args.method != 'average':
        output_filename = f"{id_part}_{args.method}_{args.components}component_results.csv"
    else:
        output_filename = f"{id_part}_{args.method}_results.csv"

    output_file_path = os.path.join(output_dir, output_filename)

    save_outputs = results_entry_to_dataframe(outputs)
    save_outputs.to_csv(
    output_file_path,
    index=False,
    sep=';')

    print(f"Results have been exported to {output_file_path}")

if __name__ == "__main__":
    main()
