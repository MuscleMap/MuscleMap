#TO DO: add descriptions to functions

import os
import logging
import sys
import json
import numpy as np
import nibabel as nib
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


def save_nifti(data, affine, filename):
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, filename)


##########################################################################################



def extract_image_data(image_path):
    img = nib.load(image_path)
    img_array = img.get_fdata()
    
    dim_x, dim_y, dim_z = img.header['dim'][1:4] #dim_z = number of axial slices
    pixdim_x, pixdim_y, pixdim_z = img.header['pixdim'][1:4] #voxel dimensions in mm

    return img, img_array, (dim_x, dim_y, dim_z), (pixdim_x, pixdim_y, pixdim_z)



def apply_clustering(mask_img, kmeans_activate, GMM_activate, components): #output_dir, image
    if kmeans_activate:
        clustering = KMeans(n_clusters = components, init = 'k-means++', tol = 0.001, n_init = 20, max_iter = 1000).fit(mask_img)
        labels = clustering.labels_ #labels is 1D, has length as number of values in mask_img, and has value corresponding to cluster assignment (0 through components-1)

    elif GMM_activate:
        clustering = GaussianMixture(n_components = components, covariance_type= 'full', init_params = 'kmeans', tol=(0.001), n_init = 20, max_iter = 1000).fit(mask_img)
        labels = clustering.predict(mask_img)
    else:
        raise ValueError("Either KMeans or GMM must be activated.")

    return labels, clustering


#labels is the label from clustering (0, 1, or 2 if 3 component)
def calculate_thresholds(labels, mask_img, num_clusters):
    clusters = [mask_img[labels == i] for i in range(num_clusters)]
    means = [np.mean(cluster) for cluster in clusters]

    # Determine upper_threshold based on the number of clusters
    if num_clusters == 2:
        muscle_max = np.max(clusters[0]) if means[0] < means[1] else np.max(clusters[1])
        muscle_img = mask_img[mask_img <= muscle_max]
        fat_img = mask_img[mask_img > muscle_max]
        unknown_img= None
        unknown_max= None
        sorted_indices = [0, 1] if means[0] < means[1] else [1, 0]
        logging.info(f"NUMCLUSTERS=2")

    elif num_clusters == 3:
        sorted_clusters = sorted(zip(means, clusters, range(len(clusters))), key=lambda x: x[0])
        muscle_img = sorted_clusters[0][1]
        unknown_img =sorted_clusters[1][1]
        fat_img = sorted_clusters[2][1]
        muscle_max = np.max(muscle_img)
        unknown_max= np.max(unknown_img)

        logging.info(f"Cluster means: {means}")
        logging.info(f"Sorted means: {sorted([mean for mean, _, _ in sorted_clusters])}")
        logging.info(f"muscle max: {np.max(muscle_img)} ")
        logging.info(f"unk min: {np.min(unknown_img)} ")
        logging.info(f"unk max: {np.max(unknown_img)}")
        logging.info(f"fat min: {np.min(fat_img)}")
        sorted_indices = [x[2] for x in sorted_clusters]

        logging.info(f"Sorted indices: {sorted_indices}")


    cluster_intensities = {
        'muscle': muscle_img,
        'unknown': unknown_img,
        'fat': fat_img
    }
    
    return muscle_max, unknown_max, cluster_intensities, sorted_indices




#this label is the muscle label
def quantify_muscle_measures(img_array, mask_array, muscle_max, unknown_max, label, sx, sy, sz):

    if unknown_max!= None: #3 component
        muscle_mask = (mask_array == label) & (img_array <= muscle_max) #muscle_mask is values where mask_array equals label AND intensity<=upper threshold
        unknown_mask = (mask_array == label) & (img_array <=unknown_max) & (img_array > muscle_max)
        fat_mask = (mask_array == label) & (img_array > unknown_max)

    else: #2 component
        muscle_mask = (mask_array == label) & (img_array <= muscle_max) 
        fat_mask = (mask_array == label) & (img_array > muscle_max)
    
    total_mask = (mask_array == label)

    muscle_percentage = round(100 * (np.sum(muscle_mask) / np.sum(total_mask)),2)
    fat_percentage = round(100 * (np.sum(fat_mask) / np.sum(total_mask)),2)

    muscle_volume_ml = round((np.sum(muscle_mask) * (sx * sy * sz)) / 1000,2)
    fat_volume_ml = round((np.sum(fat_mask) *(sx * sy * sz)) / 1000,2)
    total_volume_ml = round((np.sum(total_mask) *(sx * sy * sz)) / 1000,2)

        
    return muscle_percentage, fat_percentage, muscle_volume_ml, fat_volume_ml, total_volume_ml, muscle_mask, unknown_mask, fat_mask


def create_image_array(img_array, mask_array,label,upper_threshold):
    muscle_label = mask_array == label
    # Mask for muscle
    muscle_array = np.where(img_array < upper_threshold, muscle_label * (label), 0)
    # Mask for fat
    fat_array = np.where(img_array >= upper_threshold, muscle_label * (label), 0)
    
    return muscle_array,fat_array

def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logging.info(f"Output directory {output_dir} created")


def map_image(ID_name_file, segmentation_image, img, kmeans_activate, GMM_activate):
    if kmeans_activate:
        clustering_method="kmeans'"
    
    elif GMM_activate:
        clustering_method="GMM"

    else:
        exit()

    gt_file= f'{ID_name_file}_{clustering_method}_segmentationImage.nii.gz'
    gt_img = nib.Nifti1Image(np.rint(segmentation_image), img.affine, img.header)
    gt_img.get_data_dtype() == np.dtype(np.float64)
     # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Determine the path to the MuscleMaps directory
    musclemaps_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(musclemaps_dir, 'output')
    output_file_path = os.path.join(output_dir, gt_file)

    gt_img.to_filename(output_file_path)

