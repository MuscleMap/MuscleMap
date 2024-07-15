import os
import pandas as pd
import argparse
import numpy as np
import logging
import math
from mm_util import extract_image_data, apply_clustering,calculate_thresholds,quantify_muscle_measures,create_image_array, create_output_dir, map_image,  save_nifti

from glob import glob
import nibabel as nib



def get_parser():
    parser = argparse.ArgumentParser(description="Muscle and Fat Segmentation in MRI")

    parser.add_argument("-m", '--method', required=True, type=str, choices=['dixon', 'kmeans', 'gmm'], 
                          help="Method to use: dixon, kmeans, or gmm")

    parser.add_argument("-i", '--input', required=False, type=str, 
                          help="Input image for kmeans or gmm")
    parser.add_argument("-c", '--components', required=False, default=None, type=int, choices=[2, 3], 
                          help="Number of components for kmeans or gmm (2 or 3)")
    parser.add_argument("-s", '--segmentation_image', required=False, type=str, 
                          help="Segmentation image for any method")
    

    parser.add_argument("-f", '--fat_image', required=False, type=str, 
                          help="Fat image for dixon method")
    parser.add_argument("-w", '--water_image', required=False, type=str, 
                          help="Water image for dixon method")
    
    parser.add_argument("-o", '--output_dir', required=True, type=str, 
                          help="Output directory to save the results")
    return parser


def validate_args(args):
    if args.method == 'dixon':
        if not args.fat_image or not args.water_image or not args.segmentation_image:
            print("For dixon method, you must provide -f (fat image), -w (water image), and -s (segmentation image).")
            exit(1)
    elif args.method in ['kmeans', 'gmm']:
        if not args.input or not args.components or not args.segmentation_image:
            print("For kmeans or gmm method, you must provide -i (input image), -c (number of components), and -s (segmentation image).")
            exit(1)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    args = parser.parse_args()
    validate_args(args)



    logging.info(f"Method selected: {args.method}")
    logging.info(f"Segmentation image: {args.segmentation_image}")
    segmentations, segmentations_data, segmentations_dim,segmentations_pixdim = extract_image_data(args.segmentation_image)
    create_output_dir(args.output_dir)
    results_list=[]
    kmeans_activate=False
    GMM_activate=False

    if args.method == 'kmeans':
        kmeans_activate = True
    elif args.method == 'gmm':
        GMM_activate = True
    # Extract the ID from the input filename
    if args.method == 'dixon':
        input_filename = os.path.basename(args.fat_image)
    else:
        input_filename = os.path.basename(args.input)


    base_name = os.path.splitext(input_filename)[0]
    id_part = base_name.split('_')[0]  # This assumes the ID is the first part of the filename
    
    total_probability_maps = [np.zeros(segmentations_data.shape) for _ in range(args.components)]

    combined_muscle_mask = np.zeros(segmentations_data.shape, dtype=bool)
    combined_unknown_mask = np.zeros(segmentations_data.shape, dtype=bool)
    combined_fat_mask = np.zeros(segmentations_data.shape, dtype=bool)

    for value in np.unique(segmentations_data):
        if value>0: #iterates through muscles 
            logging.info(f"running label {value}")
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
                    image, image_array, im_dim, (sx, sy, sz) = extract_image_data(args.input) #image array has dimensions of INPUT
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
                    combined_unknown_mask |= unknown_mask
                    combined_fat_mask |= fat_mask

                    imf_percentage = np.around((fat_volume_ml / volume) * 100, decimals=2) if volume != 0 else 0 #round to 2 decimals
                    muscle_array, fat_array = create_image_array(image_array,  segmentations_data, value, muscle_max)
                    if value == 1:
                        segmentation_image = muscle_array + fat_array
                    else:
                        segmentation_image += muscle_array + fat_array
                results_list.append({
                    'Label': value, #current val iterating through, "muscle number"
                    'Volume': volume, 
                    'Number_of_slices': number_of_slices,
                    'IMF_percentage': imf_percentage,
                })
                map_image(id_part, segmentation_image,image, kmeans_activate, GMM_activate)


            # Convert results list to DataFrame
        results_df = pd.DataFrame(results_list)

    for i, component_name in enumerate(["muscle", "undefined", "fat"]):
        prob_output_filename = f'{id_part}_gmm_probabilityMask_{component_name}.nii.gz'
        prob_output_file_path = os.path.join(args.output_dir, prob_output_filename)
        prob_map_img = nib.Nifti1Image(total_probability_maps[i], image.affine, image.header)
        prob_map_img.to_filename(prob_output_file_path)
        logging.info(f"Saved total probability map for {component_name} to {prob_output_file_path}")

    save_nifti(combined_muscle_mask.astype(np.uint8), image.affine, os.path.join(args.output_dir, f'{id_part}_{args.method}_binary_muscle_mask.nii.gz'))
    save_nifti(combined_unknown_mask.astype(np.uint8), image.affine, os.path.join(args.output_dir, f'{id_part}_{args.method}_binary_undefined_mask.nii.gz'))
    save_nifti(combined_fat_mask.astype(np.uint8), image.affine, os.path.join(args.output_dir, f'{id_part}_{args.method}_binary_fat_mask.nii.gz'))


    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Determine the path to the MuscleMaps directory
    musclemaps_dir = os.path.dirname(script_dir)
    # Construct the path to the output folder
    output_dir = os.path.join(musclemaps_dir, 'output')
    output_filename = f"{id_part}_{args.method}_{args.components}_results.csv"

    # Construct the full path to the output file in the output folder
    output_file_path = os.path.join(output_dir, output_filename)

    # Export the results
    results_df.to_csv(output_file_path, index=False)
    print(f"Results have been exported to {output_file_path}")

if __name__ == "__main__":
    main()
