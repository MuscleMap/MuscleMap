import os
import pandas as pd
import argparse
import numpy as np
import logging
import math
try:
    # Attempt to import as if it is a part of a package
    from .mm_util import get_model_and_config_paths, load_model_config, validate_extract_args, extract_image_data, apply_clustering, calculate_thresholds, quantify_muscle_measures, create_image_array, create_output_dir, map_image, save_nifti, calculate_segmentation_metrics
except ImportError:
    # Fallback to direct import if run as a standalone script
    from mm_util import get_model_and_config_paths, load_model_config, validate_extract_args, extract_image_data, apply_clustering, calculate_thresholds, quantify_muscle_measures, create_image_array, create_output_dir, map_image, save_nifti, calculate_segmentation_metrics
import nibabel as nib


def get_parser():
    parser = argparse.ArgumentParser(description="Extract metrics of muscle size and composition")

    parser.add_argument("-m", '--method', required=True, type=str, choices=['dixon', 'kmeans', 'gmm', 'average'], 
                          help="Method to use: dixon, kmeans, or gmm")

    parser.add_argument("-i", '--input_image', required=False, type=str, 
                          help="Input image for kmeans or gmm")
    
    parser.add_argument("-f", '--fat_image', required=False, type=str, 
                          help="Fat image for Dixon method")

    parser.add_argument("-w", '--water_image', required=False, type=str, 
                          help="Water image for Dixon method")
    
    parser.add_argument("-s", '--segmentation_image', required=False, type=str, 
                          help="Segmentation image for any method")
    
    parser.add_argument("-c", '--components', required=False, default=None, type=int, choices=[2, 3], 
                          help="Number of components for kmeans or gmm (2 or 3)")
    
    parser.add_argument("-r", '--region', required=False, type=str,
                          help="Anatomical region. Supported regions: abdomen")

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
    segmentations, segmentations_data, segmentations_dim,segmentations_pixdim = extract_image_data(args.segmentation_image)
    output_dir=create_output_dir(args.output_dir)
    results_list=[]
    kmeans_activate=False
    GMM_activate=False

    # Get model and config paths
    if args.region:
        model_path, model_config_path = get_model_and_config_paths(args.region, None)

        # Load model configuration
        model_config = load_model_config(model_config_path)

    if args.method == 'kmeans':
        kmeans_activate = True
        total_probability_maps = None
        combined_muscle_mask = np.zeros(segmentations_data.shape, dtype=bool)
        combined_undefined_mask = np.zeros(segmentations_data.shape, dtype=bool)
        combined_fat_mask = np.zeros(segmentations_data.shape, dtype=bool)
    elif args.method == 'gmm':
        GMM_activate = True
        total_probability_maps = [np.zeros(segmentations_data.shape) for _ in range(args.components)]
        combined_muscle_mask = np.zeros(segmentations_data.shape, dtype=bool)
        combined_undefined_mask = np.zeros(segmentations_data.shape, dtype=bool)
        combined_fat_mask = np.zeros(segmentations_data.shape, dtype=bool)
    else:
        total_probability_maps = None
        combined_muscle_mask = None
        combined_undefined_mask = None
        combined_fat_mask = None

    # Extract the ID from the input filename
    if args.method == 'dixon':
        input_filename = os.path.basename(args.fat_image)
    else:
        input_filename = os.path.basename(args.input_image)
    id_part = input_filename[:-7] if input_filename.endswith('.nii.gz') else input_filename # This assumes the ID is the first part of the filename
    
    for value in np.unique(segmentations_data):

        if value>0: #iterates through muscles 
            result_entry= None
            if args.region:
                for label in model_config["labels"]:
                    if label["value"] == value:
                        structure_side_info = label["structure"] + " " + label["side"]
            else:
                structure_side_info=""
                    
            mask = segmentations_data == value #mask is when component is labelled value (0 thru whatever), has dimensions of INPUT
            #If no mask, then assign nan
            if mask.sum() == 0:
                continue
            else:
                number_of_slices = np.sum(np.max(np.max(mask, axis=0), axis=0))

                if args.method == 'dixon': 

                    metrics = calculate_segmentation_metrics(args, mask)

                elif args.method == 'average':
                    image, image_array, im_dim, im_pixdim = extract_image_data(args.input_image)
                    metrics = calculate_segmentation_metrics(args, mask)

                    pass

                elif args.method == 'kmeans' or 'gmm':

                    image, image_array, im_dim, (pixdim_x, pixdim_y, pixdim_z) = extract_image_data(args.input_image) #image array has dimensions of INPUT
                    mask_img = np.reshape(image_array[mask], (-1, 1)) #takes true positions in mask and reshapes the 1D array into 2D array with single column of input values

                    #image_array has original input dimensions, mask does too (but is boolean array). image_array[mask] gives a 1D array of intensity values for the masked regions, reshape makes it into a 2D array suitable for clustering, so mask_img is 2D

                    labels, clustering = apply_clustering(mask_img, kmeans_activate, GMM_activate, args.components) #, args.output_dir, image, id_part) #added output_dir for gmm, kmeans mask saving
                    
                    muscle_max, undefined_max, cluster_intensities, sorted_indices= calculate_thresholds(labels, mask_img, args.components)
                    
                    package= (muscle_max, undefined_max, label, pixdim_x, pixdim_y, pixdim_z)
                    #begin new segment
                    if args.method == 'gmm':

                        # Get probability maps for each component
                        probability_maps = clustering.predict_proba(mask_img) #mask_img is 2D for clustering
                        sorted_probability_maps = probability_maps[:, sorted_indices]
                        #probability_maps has dimensions (n,k), n is the number of samples (voxels) in mask_img, k is the number of components (2 or 3). so probability maps has dimensions (num of voxels, num components) with each row being a voxel, and each column being a probability of it belonging to that component

                        for i in range(args.components): 
                            prob_map = sorted_probability_maps[:, i] #returns 1D array, taking all rows and the ith column from probability_maps (a 2D array->1D array)

                            prob_map_reshaped = np.zeros(image.shape) #creates a 0 array with dimensions of original image
                            prob_map_reshaped[mask] = prob_map #prob_map_reshaped[mask] assigns the items in 3D array that belong to mask to the value from prob_map

                            # Update the total probability map for the current component
                            total_probability_maps[i] += prob_map_reshaped

                    #end of new segment
                    muscle_percentage, undefined_percentage, fat_percentage, muscle_volume, fat_volume, total_volume,  muscle_mask, undefined_mask, fat_mask = quantify_muscle_measures(image_array, segmentations_data, muscle_max, undefined_max, value, pixdim_x, pixdim_y, pixdim_z)

                    # Update cumulative masks
                    combined_muscle_mask |= muscle_mask
                    if args.components ==3:
                        combined_undefined_mask |= undefined_mask
                    combined_fat_mask |= fat_mask

                    muscle_array, fat_array = create_image_array(image_array,  segmentations_data, value, muscle_max)
                    if value == 1:
                        segmentation_image = muscle_array + fat_array
                    else:
                        segmentation_image += muscle_array + fat_array
                    results = {}
                    results['Muscle (%)'] = muscle_percentage
                    results['Fat (%)'] = fat_percentage
                    if args.components == 3:
                        results['Undefined (%)'] = undefined_percentage
                    results['Volume (ml)'] = total_volume
                    metrics=results

                result_entry={
                    'Structure': structure_side_info.strip(),
                    'Label': value, #current val iterating through, "muscle number",
                    'Number of Slices': number_of_slices,
                }
                result_entry.update(metrics)
                results_list.append(result_entry)
        
        results_df = pd.DataFrame(results_list)

    if args.method == 'gmm':
        if args.components == 3:
            array=["muscle", "undefined", "fat"]
        else:
            array=["muscle", "fat"]
        for i, component_name in enumerate(array):
            prob_output_filename = f'{id_part}_gmm_{component_name}_{args.components}component_softseg.nii.gz'
            prob_output_file_path = os.path.join(output_dir, prob_output_filename)
            prob_map_img = nib.Nifti1Image(total_probability_maps[i], image.affine, image.header)
            prob_map_img.to_filename(prob_output_file_path)
            logging.info(f"Saved total probability map for {component_name} to {prob_output_file_path}")

    if args.method != 'dixon' and args.method != 'average':
        save_nifti(combined_muscle_mask.astype(np.uint8), image.affine, os.path.join(output_dir, f'{id_part}_{args.method}_{args.components}component_muscle_seg.nii.gz'))
        if args.components == 3:
            save_nifti(combined_undefined_mask.astype(np.uint8), image.affine, os.path.join(output_dir, f'{id_part}_{args.method}_{args.components}component_undefined_seg.nii.gz'))
        save_nifti(combined_fat_mask.astype(np.uint8), image.affine, os.path.join(output_dir, f'{id_part}_{args.method}_{args.components}component_fat_seg.nii.gz'))

    # Construct the path to the output CSV file
    if args.method != 'dixon' and args.method != 'average':
        output_filename = f"{id_part}_{args.method}_{args.components}component_results.csv"
    else:
        output_filename = f"{id_part}_{args.method}_results.csv"

    output_file_path = os.path.join(output_dir, output_filename)

    # Export the results
    results_df.to_csv(output_file_path, index=False)
    print(f"Results have been exported to {output_file_path}")

if __name__ == "__main__":
    main()
