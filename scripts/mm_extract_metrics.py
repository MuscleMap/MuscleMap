import os
import sys
import pandas as pd
import argparse
import numpy as np
import logging
from mm_Util import extract_image_data, apply_clustering,calculate_thresholds,quantify_muscle_measures,extract_ID_file_name,create_image_array,map_image,create_excel

def get_parser():
    parser = argparse.ArgumentParser(description="Muscle and Fat Segmentation in Lumbar Spine MRI")

    parser.add_argument("-m", '--method', required=True, type=str, choices=['dixon', 'kmeans', 'gmm'], 
                          help="Method to use: dixon, kmeans, or gmm")

    parser.add_argument("-i", '--input', required=False, type=str, 
                          help="Input image for kmeans or gmm")
    parser.add_argument("-c", '--components', required=False, type=int, choices=[2, 3], 
                          help="Number of components for kmeans or gmm (2 or 3)")
    parser.add_argument("-s", '--segmentation_image', required=False, type=str, 
                          help="Segmentation image for any method")
    
    parser.add_argument("-f", '--fat_image', required=False, type=str, 
                          help="Fat image for dixon method")
    parser.add_argument("-w", '--water_image', required=False, type=str, 
                          help="Water image for dixon method")

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

    kmeans_activate = False
    GMM_activate = False

    if args.method == 'kmeans':
        kmeans_activate = True
    elif args.method == 'gmm':
        GMM_activate = True


    logging.info(f"Method selected: {args.method}")
    
    if args.method == 'dixon':
        logging.info(f"Fat image: {args.fat_image}")
        logging.info(f"Water image: {args.water_image}")
        logging.info(f"Segmentation image: {args.segmentation_image}")
        # Add your dixon method processing code here
    elif args.method in ['kmeans', 'gmm']:
        logging.info(f"Input image: {args.input}")
        logging.info(f"Number of components: {args.components}")
        logging.info(f"Segmentation image: {args.segmentation_image}")
        # Add your kmeans or gmm method processing code here
    






    #Find all images and masks
    image_path = args.input
    mask_path = args.segmentation_image


    
    #empty lists for each variable per muscle 
    muscle_multifidus_right, muscle_multifidus_left = [], []
    muscle_erector_right, muscle_erector_left = [], []
    muscle_psoas_right, muscle_psoas_left = [], []
    fat_multifidus_right, fat_multifidus_left = [], []
    fat_erector_right, fat_erector_left = [], []
    fat_psoas_right, fat_psoas_left = [], []
    volume_total_multifidus_right, volume_total_multifidus_left = [], []
    volume_total_erector_right, volume_total_erector_left = [], []
    volume_total_psoas_right, volume_total_psoas_left = [], []
    volume_muscle_multifidus_right, volume_muscle_multifidus_left = [], []
    volume_muscle_erector_right, volume_muscle_erector_left = [], []
    volume_muscle_psoas_right, volume_muscle_psoas_left = [], []
    volume_fat_multifidus_right, volume_fat_multifidus_left = [], []
    volume_fat_erector_right, volume_fat_erector_left = [], []
    volume_fat_psoas_right, volume_fat_psoas_left = [], []
    ID_name = []
    
    #loop over folder
    # Process single input and segmentation image
    ID_name_file = extract_ID_file_name(image_path)
    ID_name.append(ID_name_file)
    # Extract image data
    img, img_array, mask_array, (sx, sy, sz) = extract_image_data(image_path, mask_path)

    # Process each muscle group
    for label_pair, muscle_group in zip([(1, 2), (3, 4), (5, 6)], ['multifidus', 'erector', 'psoas']):
        combined_mask = np.logical_or(mask_array == label_pair[0], mask_array == label_pair[1])
        mask_img = np.reshape(img_array[combined_mask], (-1, 1))
        #loop over de muscles
        if muscle_group == 'multifidus':
            labels_multifidus = apply_clustering(mask_img, kmeans_activate, GMM_activate, args.components)
            upper_threshold, muscle_image, fat_img  = calculate_thresholds(labels_multifidus,mask_img) 
            # loop over the sides
            for side in (['right', 'left']):
                if side == 'right':
                    label = 1
                    #use quantify function to calculate the measures
                    muscle_percentage, fat_percentage, muscle_volume_ml, fat_volume_ml, total_volume_ml = quantify_muscle_measures(img_array, mask_array, upper_threshold, label, sx, sy, sz)
                
                    #append outcomes from quantify function to lists
                    muscle_multifidus_right.append(muscle_percentage)
                    fat_multifidus_right.append(fat_percentage)
                    volume_muscle_multifidus_right.append(muscle_volume_ml)
                    volume_fat_multifidus_right.append(fat_volume_ml)
                    volume_total_multifidus_right.append(total_volume_ml)  
                
                    # Mask for muscle
                    muscle_array,fat_array = create_image_array(img_array, mask_array,label,upper_threshold)
                    segmentation_image =  muscle_array + fat_array
                    
                if side == 'left':
                    label = 2
                    #use quantify function to calculate the measures
                    muscle_percentage, fat_percentage, muscle_volume_ml, fat_volume_ml, total_volume_ml = quantify_muscle_measures(img_array, mask_array, upper_threshold, label, sx, sy, sz)
                    
                    #append outcomes from quantify function to lists
                    muscle_multifidus_left.append(muscle_percentage)
                    fat_multifidus_left.append(fat_percentage)
                    volume_muscle_multifidus_left.append(muscle_volume_ml)
                    volume_fat_multifidus_left.append(fat_volume_ml)
                    volume_total_multifidus_left.append(total_volume_ml)  
                
                    # Mask for muscle
                    muscle_array,fat_array = create_image_array(img_array, mask_array,label,upper_threshold)
                    segmentation_image = segmentation_image+ muscle_array + fat_array
                    
        if muscle_group == 'erector':
            labels_erector = apply_clustering(mask_img, kmeans_activate, GMM_activate, args.components)
            upper_threshold, muscle_image, fat_img  = calculate_thresholds(labels_erector,mask_img) 
            # loop over the sides
            for side in (['right', 'left']):
                if side == 'right':
                    label = 3
                    #use quantify function to calculate the measures
                    muscle_percentage, fat_percentage, muscle_volume_ml, fat_volume_ml, total_volume_ml = quantify_muscle_measures(img_array, mask_array, upper_threshold, label, sx, sy, sz)
                
                    #append outcomes from quantify function to lists
                    muscle_erector_right.append(muscle_percentage)
                    fat_erector_right.append(fat_percentage)
                    volume_muscle_erector_right.append(muscle_volume_ml)
                    volume_fat_erector_right.append(fat_volume_ml)
                    volume_total_erector_right.append(total_volume_ml)  
                
                    # Mask for muscle
                    muscle_array,fat_array = create_image_array(img_array, mask_array,label,upper_threshold)
                    segmentation_image = segmentation_image + muscle_array + fat_array
                if side == 'left':
                    label = 4
                    #use quantify function to calculate the measures
                    muscle_percentage, fat_percentage, muscle_volume_ml, fat_volume_ml, total_volume_ml = quantify_muscle_measures(img_array, mask_array, upper_threshold, label, sx, sy, sz)
                    
                    #append outcomes from quantify function to lists
                    muscle_erector_left.append(muscle_percentage)
                    fat_erector_left.append(fat_percentage)
                    volume_muscle_erector_left.append(muscle_volume_ml)
                    volume_fat_erector_left.append(fat_volume_ml)
                    volume_total_erector_left.append(total_volume_ml)  
                
                    # Mask for muscle
                    muscle_array,fat_array = create_image_array(img_array, mask_array,label,upper_threshold)
                    segmentation_image = segmentation_image+ muscle_array + fat_array       
        if muscle_group == 'psoas':
            labels_psoas = apply_clustering(mask_img, kmeans_activate, GMM_activate, args.components)
            upper_threshold, muscle_image, fat_img  = calculate_thresholds(labels_psoas,mask_img) 
            # loop over the sides
            for side in (['right', 'left']):
                if side == 'right':
                    label = 5
                    #use quantify function to calculate the measures
                    muscle_percentage, fat_percentage, muscle_volume_ml, fat_volume_ml, total_volume_ml = quantify_muscle_measures(img_array, mask_array, upper_threshold, label, sx, sy, sz)
                
                    #append outcomes from quantify function to lists
                    muscle_psoas_right.append(muscle_percentage)
                    fat_psoas_right.append(fat_percentage)
                    volume_muscle_psoas_right.append(muscle_volume_ml)
                    volume_fat_psoas_right.append(fat_volume_ml)
                    volume_total_psoas_right.append(total_volume_ml)  
                
                    # Mask for muscle
                    muscle_array,fat_array = create_image_array(img_array, mask_array,label,upper_threshold)
                    segmentation_image = segmentation_image + muscle_array + fat_array
                    
                if side == 'left':
                    label = 6
                    #use quantify function to calculate the measures
                    muscle_percentage, fat_percentage, muscle_volume_ml, fat_volume_ml, total_volume_ml = quantify_muscle_measures(img_array, mask_array, upper_threshold, label, sx, sy, sz)
                    
                    #append outcomes from quantify function to lists
                    muscle_psoas_left.append(muscle_percentage)
                    fat_psoas_left.append(fat_percentage)
                    volume_muscle_psoas_left.append(muscle_volume_ml)
                    volume_fat_psoas_left.append(fat_volume_ml)
                    volume_total_psoas_left.append(total_volume_ml)  
                
                    # Mask for muscle
                    muscle_array,fat_array = create_image_array(img_array, mask_array,label,upper_threshold)
                    segmentation_image = segmentation_image+ muscle_array + fat_array
        
                    
        map_image (ID_name_file, segmentation_image,img,"GMM" if GMM_activate else "K-means")
            
    list_muscle = pd.DataFrame([ID_name, muscle_multifidus_right, muscle_multifidus_left, muscle_erector_right, muscle_erector_left, muscle_psoas_right, muscle_psoas_left]).T
    list_fat = pd.DataFrame([ID_name, fat_multifidus_right, fat_multifidus_left, fat_erector_right, fat_erector_left, fat_psoas_right, fat_psoas_left]).T
    list_volume = pd.DataFrame([ID_name, volume_total_multifidus_right, volume_total_multifidus_left, volume_total_erector_right, volume_total_erector_left, volume_total_psoas_right, volume_total_psoas_left]).T
    list_muscle_volume = pd.DataFrame([ID_name, volume_muscle_multifidus_right, volume_muscle_multifidus_left, volume_muscle_erector_right, volume_muscle_erector_left, volume_muscle_psoas_right, volume_muscle_psoas_left]).T
    list_fat_volume = pd.DataFrame([ID_name, volume_fat_multifidus_right, volume_fat_multifidus_left, volume_fat_erector_right, volume_fat_erector_left, volume_fat_psoas_right, volume_fat_psoas_left]).T
    lists = [list_muscle, list_fat, list_volume, list_muscle_volume, list_fat_volume]
    
    file_name = f"Muscle_Measures_{'GMM' if GMM_activate else 'K-means'}.xlsx"
    sheet_names = ['Muscle(%)', 'IMF(%)', 'Muscle(ml)', 'IMF(ml)', 'IMF(ml)']
    titles = ['LMR', 'LML', 'ESR', 'ESL', 'PMR', 'PML']
    
    create_excel(lists, file_name, sheet_names, titles)




if __name__ == "__main__":
    main()
