import os
import pandas as pd
import argparse
import numpy as np
import logging
import math
try:
    # Attempt to import as if it is a part of a package
    from .mm_util import get_model_and_config_paths, load_model_config, validate_extract_args, extract_image_data, apply_clustering, calculate_thresholds, quantify_muscle_measures, create_image_array, create_output_dir, map_image, save_nifti
except ImportError:
    # Fallback to direct import if run as a standalone script
    from mm_util import get_model_and_config_paths, load_model_config, validate_extract_args, extract_image_data, apply_clustering, calculate_thresholds, quantify_muscle_measures, create_image_array, create_output_dir, map_image, save_nifti
import nibabel as nib

def get_parser():
    parser = argparse.ArgumentParser(description="Muscle and Fat Segmentation in MRI")

    parser.add_argument("-m", '--method', required=True, type=str, choices=['dixon', 'kmeans', 'gmm'], 
                          help="Method to use: dixon, kmeans, or gmm")

    parser.add_argument("-i", '--input_image', required=False, type=str, 
                          help="Input image for kmeans or gmm")
    

    parser.add_argument("-f", '--fat_image', required=False, type=str, 
                          help="Fat image for dixon method")
    parser.add_argument("-w", '--water_image', required=False, type=str, 
                          help="Water image for dixon method")
    

    parser.add_argument("-s", '--segmentation_image', required=False, type=str, 
                          help="Segmentation image for any method")
    parser.add_argument("-c", '--components', required=False, default=None, type=int, choices=[2, 3], 
                          help="Number of components for kmeans or gmm (2 or 3)")
    
    parser.add_argument("-r", '--region', required=False, type=str,
                          help="output name.")



    parser.add_argument("-o", '--output_dir', required=False, type=str, 
                          help="Output directory to save the results, can be absolute or relative")
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
        combined_unknown_mask = np.zeros(segmentations_data.shape, dtype=bool)
        combined_fat_mask = np.zeros(segmentations_data.shape, dtype=bool)

    elif args.method == 'gmm':
        GMM_activate = True
        total_probability_maps = [np.zeros(segmentations_data.shape) for _ in range(args.components)]
        combined_muscle_mask = np.zeros(segmentations_data.shape, dtype=bool)
        combined_unknown_mask = np.zeros(segmentations_data.shape, dtype=bool)
        combined_fat_mask = np.zeros(segmentations_data.shape, dtype=bool)
    else:
        total_probability_maps = None
        combined_muscle_mask = None
        combined_unknown_mask = None
        combined_fat_mask = None

    # Extract the ID from the input filename
    if args.method == 'dixon':
        input_filename = os.path.basename(args.fat_image)
    else:
        input_filename = os.path.basename(args.input_image)
    id_part = input_filename[:-7] if input_filename.endswith('.nii.gz') else input_filename # This assumes the ID is the first part of the filename
    
    for value in np.unique(segmentations_data):
        if value>0: #iterates through muscles 
            logging.info(f"running label {value}")
            if args.region:
                for label in model_config["labels"]:
                    if label["value"] == value:
                        muscle_side_info = label["muscle"] + " " + label["side"]
            else:
                muscle_side_info=""
                    
            mask = segmentations_data == value #mask is when component is labelled value (0 thru whatever), has dimensions of INPUT
            #If no mask, then assign nan
            if mask.sum() == 0:
                volume = np.nan
                number_of_slices = np.nan
                imf_percentage = np.nan
            else:
                number_of_slices = np.sum(np.max(np.max(mask, axis=0), axis=0))
                if args.method == 'dixon':     
                    logging.info(f"Fat image: {args.fat_image}")
                    logging.info(f"Water image: {args.water_image}")
                    water, water_array, water_dim, water_pixdim = extract_image_data(args.water_image)
                    fat, fat_array, fat_dim, fat_pixdim = extract_image_data(args.fat_image)
                    volume=np.around(mask.sum() * math.prod(water_pixdim)/ 1000, decimals=2)
                    imf_percentage = np.around((fat_array[mask].sum() / (fat_array[mask].sum() + water_array[mask].sum())) * 100, decimals=2)

                elif args.method == 'kmeans' or 'gmm':
                    image, image_array, im_dim, (sx, sy, sz) = extract_image_data(args.input_image) #image array has dimensions of INPUT
                    mask_img = np.reshape(image_array[mask], (-1, 1)) #takes true positions in mask and reshapes the 1D array into 2D array with single column of input values

                    #image_array has original input dimensions, mask does too (but is boolean array). image_array[mask] gives a 1D array of intensity values for the masked regions, reshape makes it into a 2D array suitable for clustering, so mask_img is 2D

                    labels, clustering = apply_clustering(mask_img, kmeans_activate, GMM_activate, args.components) #, args.output_dir, image, id_part) #added output_dir for gmm, kmeans mask saving
                    
                    muscle_max, unknown_max, cluster_intensities, sorted_indices= calculate_thresholds(labels, mask_img, args.components)
                    
                    #begin new segment
                    if args.method == 'gmm':
                        logging.info(f"Input image dimensions: {image.shape}")

                        # Get probability maps for each component
                        probability_maps = clustering.predict_proba(mask_img) #mask_img is 2D for clustering
                        sorted_probability_maps = probability_maps[:, sorted_indices]
                        #probability_maps has dimensions (n,k), n is the number of samples (voxels) in mask_img, k is the number of components (2 or 3). so probability maps has dimensions (num of voxels, num components) with each row being a voxel, and each column being a probability of it belonging to that component

                        for i in range(args.components): 
                            prob_map = sorted_probability_maps[:, i] #returns 1D array, taking all rows and the ith column from probability_maps (a 2D array->1D array)
                            logging.info(f"Dimensions of prob_map for component {i}: {prob_map.shape}") 

                            prob_map_reshaped = np.zeros(image.shape) #creates a 0 array with dimensions of original image
                            prob_map_reshaped[mask] = prob_map #prob_map_reshaped[mask] assigns the items in 3D array that belong to mask to the value from prob_map
                            logging.info(f"Dimensions of prob_map_reshaped for component {i}: {prob_map_reshaped.shape}")

                            # Update the total probability map for the current component
                            total_probability_maps[i] += prob_map_reshaped
                            logging.info(f"Updated total_probability_map for component {i}")

                    #end of new segment
                    muscle_percentage, fat_percentage, muscle_volume_ml, fat_volume_ml, volume,  muscle_mask, unknown_mask, fat_mask = quantify_muscle_measures(image_array, segmentations_data, muscle_max, unknown_max, value, sx, sy, sz)

                    # Update cumulative masks
                    combined_muscle_mask |= muscle_mask
                    if args.components ==3:
                        combined_unknown_mask |= unknown_mask
                    combined_fat_mask |= fat_mask

                    imf_percentage = np.around((fat_volume_ml / volume) * 100, decimals=2) if volume != 0 else 0 #round to 2 decimals
                    muscle_array, fat_array = create_image_array(image_array,  segmentations_data, value, muscle_max)
                    if value == 1:
                        segmentation_image = muscle_array + fat_array
                    else:
                        segmentation_image += muscle_array + fat_array
                    
                results_list.append({
                    'muscle': muscle_side_info,
                    'Label': value, #current val iterating through, "muscle number",
                    'Volume (mL)': volume, 
                    'Number_of_slices': number_of_slices,
                    'IMF_percentage': imf_percentage,

                })
                #if args.method =="kmeans" or 'gmm':
                #map_image(id_part, segmentation_image,image, kmeans_activate, GMM_activate)

            # Convert results list to DataFrame
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

    if args.method != 'dixon':
        save_nifti(combined_muscle_mask.astype(np.uint8), image.affine, os.path.join(output_dir, f'{id_part}_{args.method}_{args.components}component_muscle_seg.nii.gz'))
        if args.components == 3:
            save_nifti(combined_unknown_mask.astype(np.uint8), image.affine, os.path.join(output_dir, f'{id_part}_{args.method}_{args.components}component_undefined_seg.nii.gz'))
        save_nifti(combined_fat_mask.astype(np.uint8), image.affine, os.path.join(output_dir, f'{id_part}_{args.method}_{args.components}component_fat_seg.nii.gz'))

    # Construct the path to the output CSV file
    
    output_filename = f"{id_part}_{args.method}_{args.components}component_results.csv"
    output_file_path = os.path.join(output_dir, output_filename)

    # Export the results
    results_df.to_csv(output_file_path, index=False)
    print(f"Results have been exported to {output_file_path}")

if __name__ == "__main__":
    main()
