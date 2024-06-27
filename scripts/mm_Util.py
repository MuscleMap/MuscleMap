#TO DO: add descriptions to functions

import os
import logging
import sys
import json
import numpy as np
import nibabel as nib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

#check_image_exists (DESCRIPTION)
def check_image_exists(image_path):
    if not os.path.isfile(image_path):
        logging.error(f"Image file '{image_path}' does not exist or is not a file.")
        sys.exit(1)
    if not os.access(image_path, os.R_OK):
        logging.error(f"Image file '{image_path}' is not readable.")
        sys.exit(1)


def get_model_and_config_paths(region, specified_model=None):
    models_base_dir = os.path.join(os.path.dirname(__file__), "..", "models", region)
    
    if specified_model:
        model_path = os.path.join(models_base_dir, specified_model)
        config_path = os.path.splitext(model_path)[0] + ".json"
        if not os.path.isfile(model_path):
            logging.error(f"Specified model '{specified_model}' does not exist.")
            sys.exit(1)
        if not os.path.isfile(config_path):
            logging.error(f"Config file for model '{specified_model}' does not exist.")
            sys.exit(1)
    else:
        if not os.path.isdir(models_base_dir):
            logging.error(f"Region folder '{region}' does not exist.")
            sys.exit(1)
        
        # Assuming only one model file and one config file in each region folder
        model_path = None
        config_path = None

        for file in os.listdir(models_base_dir):
            if file.endswith(".pth"):
                model_path = os.path.join(models_base_dir, file)
            elif file.endswith(".json"):
                config_path = os.path.join(models_base_dir, file)

        if not model_path:
            logging.error(f"No model file found in region folder '{region}'.")
            sys.exit(1)
        if not config_path:
            logging.error(f"No config file found in region folder '{region}'.")
            sys.exit(1)

    return model_path, config_path




def load_model_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        logging.error(f"Error: The configuration file '{config_path}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        logging.error(f"Error parsing the configuration file: {exc}")
        sys.exit(1)


def validate_arguments(args):
    if not args.images:
        logging.error("Error: The input image (-i) argument is required.")
        sys.exit(1)
    if args.images and not isinstance(args.images, str):
        logging.error("Error: The input image (-i) argument must be a string.")
        sys.exit(1)    
    
    if not args.region:
        logging.error("Error: The body region (-r) argument is required.")
        sys.exit(1)
    if args.region and not isinstance(args.region, str):
        logging.error("Error: The body region (-r) argument must be a string.")
        sys.exit(1)  

    # Optional Argument input=type string validation
    if args.model and not isinstance(args.model, str):
        logging.error("Error: The model (-m) argument must be a string.")
        sys.exit(1)
    
    if args.output_file_name and not isinstance(args.output_file_name, str):
        logging.error("Error: The output file name (-o) argument must be a string.")
        sys.exit(1)
    if args.output_dir and not isinstance(args.output_dir, str):
        logging.error("Error: The output directory (-s) argument must be a string.")
        sys.exit(1)


##########################################################################################



def extract_image_data(image_path, mask_path):
    img = nib.load(image_path)
    img_array = img.get_fdata()
    mask = nib.load(mask_path)
    mask_array = mask.get_fdata()

    hdr = img.header
    zooms = hdr.get_zooms()
    sx, sy, sz = zooms

    return img, img_array, mask_array, (sx, sy, sz)

def apply_clustering(mask_img, kmeans_activate, GMM_activate, components):
    if kmeans_activate:
        clustering = KMeans(n_clusters = components, init = 'k-means++', tol = 0.001, n_init = 20, max_iter = 1000).fit(mask_img)
        labels = clustering.labels_
    elif GMM_activate:
        clustering = GaussianMixture(n_components = components, covariance_type= 'full', init_params = 'kmeans', tol=(0.001), n_init = 20, max_iter = 1000).fit(mask_img)
        labels = clustering.predict(mask_img)
    else:
        raise ValueError("Either KMeans or GMM must be activated.")

    return labels

def calculate_thresholds(labels, mask_img):
    cluster1 = mask_img[labels == 0]
    cluster2 = mask_img[labels == 1]

    mean1 = np.mean(cluster1)
    mean2 = np.mean(cluster2)

    upper_threshold = np.max(cluster1) if mean1 < mean2 else np.max(cluster2)
    muscle_img = mask_img[mask_img <= upper_threshold]
    fat_img = mask_img[mask_img > upper_threshold]

    return upper_threshold, muscle_img, fat_img

def quantify_muscle_measures(img_array, mask_array, upper_threshold, label, sx, sy, sz):
    muscle_mask = (mask_array == label) & (img_array <= upper_threshold)
    fat_mask = (mask_array == label) & (img_array > upper_threshold)
    total_mask = (mask_array == label)

    muscle_percentage = round(100 * (np.sum(muscle_mask) / np.sum(total_mask)),2)
    fat_percentage = round(100 * (np.sum(fat_mask) / np.sum(total_mask)),2)

    muscle_volume_ml = round((np.sum(muscle_mask) * (sx * sy * sz)) / 1000,2)
    fat_volume_ml = round((np.sum(fat_mask) *(sx * sy * sz)) / 1000,2)
    total_volume_ml = round((np.sum(total_mask) *(sx * sy * sz)) / 1000,2)
    
    return muscle_percentage, fat_percentage, muscle_volume_ml, fat_volume_ml, total_volume_ml
    return muscle_percentage, fat_percentage, muscle_volume_ml, fat_volume_ml, total_volume_ml


def extract_ID_file_name(mask_path):
    # Extract the directory name containing 'testing'
    dir_name = os.path.dirname(mask_path)
    
    # Find the index where 'testing' occurs
    index = dir_name.find('testing')
    index= index + 8  
    
    # Extract the substring starting from 6 characters after 'testing'
    ID_file_name = mask_path[index:index+6]
    
    return ID_file_name

def create_image_array(img_array, mask_array,label,upper_threshold):
    muscle_label = mask_array == label
    # Mask for muscle
    muscle_array = np.where(img_array < upper_threshold, muscle_label * (label*10+1), 0)
    # Mask for fat
    fat_array = np.where(img_array >= upper_threshold, muscle_label * (label*10+2), 0)
    
    return muscle_array,fat_array

def map_image(ID_name_file, segmentation_image, img, clustering_method):
    gt_file= f'{ID_name_file}_Bimodal_{clustering_method}_GT.nii.gz'
    gt_img = nib.Nifti1Image(np.rint(segmentation_image), img.affine, img.header)
    gt_img.get_data_dtype() == np.dtype(np.float64)
    gt_img.to_filename(gt_file)
 
def create_excel(lists, file_name, sheet_names, titles):
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        workbook = writer.book

        # Define formats
        bold_green = workbook.add_format({'bold': True, 'font_color': 'green'})
        tab_color_muscle = 'red'
        tab_color_imf = 'orange'

        for i, df in enumerate(lists):
            df.columns = ['ID_name'] + titles
            df.to_excel(writer, sheet_name=sheet_names[i], index=False)

            # Get the current worksheet
            worksheet = writer.sheets[sheet_names[i]]

            # Set tab color
            if 'Muscle' in sheet_names[i]:
                worksheet.set_tab_color(tab_color_muscle)
            else:
                worksheet.set_tab_color(tab_color_imf)

            # Write headers with bold green format
            for j, title in enumerate(['ID_name'] + titles):
                worksheet.write(0, j, title, bold_green)
