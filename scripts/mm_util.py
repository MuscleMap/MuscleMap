import os
import logging
import sys
import json
import numpy as np
import nibabel as nib
import math
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from monai.transforms import ( MapTransform)
import gc, torch
from contextlib import nullcontext
import os, gc, torch, nibabel as nib
import torch.nn.functional as F

#check_image_exists 
def check_image_exists(image_path):
    if not os.path.isfile(image_path):
        logging.error(f"Image file '{image_path}' does not exist or is not a file.")
        sys.exit(1)
    if not os.access(image_path, os.R_OK):
        logging.error(f"Image file '{image_path}' is not readable.")
        sys.exit(1)

def get_model_and_config_paths(region, specified_model=None):
    models_base_dir = os.path.join(os.path.dirname(__file__), "models", region)
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

def get_template_paths(region, specified_template=None):
    templates_base_dir = os.path.join(os.path.dirname(__file__), "templates", region)
    
    if not os.path.isdir(templates_base_dir):
        logging.error(f"Region folder '{region}' does not exist.")
        sys.exit(1)
    
    print(templates_base_dir)

    if specified_template:
        template_path = os.path.join(templates_base_dir, specified_template + '.nii.gz')
        template_segmentation_path = os.path.join(templates_base_dir, specified_template + '_dseg.nii.gz')
    else:
        template_path = os.path.join(templates_base_dir, region + '_template.nii.gz')
        template_segmentation_path = os.path.join(templates_base_dir, region + '_template_dseg.nii.gz')
        
    
        if not os.path.isfile(template_path):
            logging.error(f"No template file found in region folder '{region}': ${template_path}.")
            sys.exit(1)
            
        if not os.path.isfile(template_segmentation_path):
            logging.error(f"No template segmentation file found in region folder '{region}': ${template_segmentation_path}.")
            sys.exit(1)

    return template_path, template_segmentation_path


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


def validate_seg_arguments(args):
    required_args = {'input_image': "-i", 'region': "-r"}
    for arg_name, flag in required_args.items():
        arg_value = getattr(args, arg_name, None)
        if not arg_value:
            logging.error(f"Error: The {arg_name} ({flag}) argument is required.")
            sys.exit(1)
        if not isinstance(arg_value, str):
            logging.error(f"Error: The {arg_name} ({flag}) argument must be a string.")
            sys.exit(1)

    # Optional arguments validation
    optional_args = {'model': "-m"}
    for arg_name, flag in optional_args.items():
        arg_value = getattr(args, arg_name, None)
        if arg_value and not isinstance(arg_value, str):
            logging.error(f"Error: The {arg_name} ({flag}) argument must be a string.")
            sys.exit(1)

def save_nifti(data, affine, filename):
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, filename)

def validate_extract_args(args):
    if args.method == 'dixon':
        if not args.fat_image or not args.water_image or not args.segmentation_image:
            print("For dixon method, you must provide -f (fat image), -w (water image), and -s (segmentation image).")
            exit(1)
    elif args.method in ['kmeans', 'gmm']:
        if not args.input_image or not args.components or not args.segmentation_image:
            print("For kmeans or gmm method, you must provide -i (input image), -c (number of components), and -s (segmentation image).")
            exit(1)
    elif args.method == 'average':
        if not args.input_image or not args.segmentation_image:
            print("For average, you must provide -i (input image) and -s (segmentation image).")
    string_args = ['fat_image', 'water_image', 'segmentation_image', 'input_image', 'region', 'output_dir']
    for arg_name in string_args:
        arg_value = getattr(args, arg_name, None)
        if arg_value and not isinstance(arg_value, str):
            logging.error(f"Error: The {arg_name} argument must be a string.")
            sys.exit(1)

def validate_register_to_template_args(args):
    if not args.input_image or not args.components or not args.segmentation_image:
        print("You must provide -i (input image), -s (segmentation image), and -r (region).")
        exit(1)
    string_args = ['input_image', 'segmentation_image', 'region', 'output_dir']
    for arg_name in string_args:
        arg_value = getattr(args, arg_name, None)
        if arg_value and not isinstance(arg_value, str):
            logging.error(f"Error: The {arg_name} argument must be a string.")
            sys.exit(1)

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
        clustering = GaussianMixture(n_components = components, covariance_type = 'full', init_params = 'kmeans', tol = 0.001, n_init = 20, max_iter = 1000).fit(mask_img)
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
        undefined_img= None
        undefined_max= None
        sorted_indices = [0, 1] if means[0] < means[1] else [1, 0]

    elif num_clusters == 3:
        sorted_clusters = sorted(zip(means, clusters, range(len(clusters))), key=lambda x: x[0])
        muscle_img = sorted_clusters[0][1]
        undefined_img =sorted_clusters[1][1]
        fat_img = sorted_clusters[2][1]
        muscle_max = np.max(muscle_img)
        undefined_max= np.max(undefined_img)

        sorted_indices = [x[2] for x in sorted_clusters]

    cluster_intensities = {
        'muscle': muscle_img,
        'undefined': undefined_img,
        'fat': fat_img
    }
    
    return muscle_max, undefined_max, cluster_intensities, sorted_indices


#this label is the muscle label
def quantify_muscle_measures(img_array, mask_array, muscle_max, undefined_max, label, pixdim_x, pixdim_y, pixdim_z):
    total_mask = (mask_array == label)

    if undefined_max!= None: #3 component
        muscle_mask = (mask_array == label) & (img_array <= muscle_max) #muscle_mask is values where mask_array equals label AND intensity<=upper threshold
        undefined_mask = (mask_array == label) & (img_array <=undefined_max) & (img_array > muscle_max)
        fat_mask = (mask_array == label) & (img_array > undefined_max)
        undefined_percentage = round(100 * (np.sum(undefined_mask) / np.sum(total_mask)),2)


    else: #2 component
        muscle_mask = (mask_array == label) & (img_array <= muscle_max) 
        fat_mask = (mask_array == label) & (img_array > muscle_max)
        undefined_mask=None
        undefined_percentage=None
    
    muscle_percentage = round(100 * (np.sum(muscle_mask) / np.sum(total_mask)),2)
    fat_percentage = round(100 * (np.sum(fat_mask) / np.sum(total_mask)),2)
    muscle_volume = round((np.sum(muscle_mask) * (pixdim_x * pixdim_y * pixdim_z)) / 1000,2) #Volume in ml
    fat_volume = round((np.sum(fat_mask) *(pixdim_x * pixdim_y * pixdim_z)) / 1000,2) #Volume in ml
    total_volume = round((np.sum(total_mask) *(pixdim_x * pixdim_y * pixdim_z)) / 1000,2) #Volume in ml
        
    return muscle_percentage, undefined_percentage, fat_percentage, muscle_volume, fat_volume, total_volume, muscle_mask, undefined_mask, fat_mask


def create_image_array(img_array, mask_array,label,upper_threshold):
    muscle_label = mask_array == label
    # Mask for muscle
    muscle_array = np.where(img_array < upper_threshold, muscle_label * (label), 0)
    # Mask for fat
    fat_array = np.where(img_array >= upper_threshold, muscle_label * (label), 0)
    
    return muscle_array, fat_array


def create_output_dir(output_dir=None):
    if not output_dir:
        output_dir = os.getcwd()  # Use the current working directory if no output directory is provided
    
    else:
        # Construct the path to the output directory from the current working directory
        output_dir = os.path.abspath(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Output directory {output_dir} created")

    return output_dir


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


def calculate_segmentation_metrics(args, mask):
    results = {}
    if args.method == 'dixon':
        water, water_array, water_dim, water_pixdim = extract_image_data(args.water_image)
        fat, fat_array, fat_dim, fat_pixdim = extract_image_data(args.fat_image)
        volume = np.around(mask.sum() * math.prod(water_pixdim) / 1000, decimals=2)
        imf_percentage = np.around((fat_array[mask].sum() / (fat_array[mask].sum() + water_array[mask].sum())) * 100, decimals=2)
        results['Volume (ml)'] = volume
        results['IMF (%)'] = imf_percentage

    elif args.method == 'average':
        image, image_array, im_dim, im_pixdim = extract_image_data(args.input_image)

        masked_image_array = image_array[mask]  # Apply mask to image array
        average_intensity = np.around(np.mean(masked_image_array), decimals=3)
        volume = np.around(mask.sum() * np.prod(im_pixdim) / 1000, decimals=3)  # Convert from mm^3 to mL
        results['Volume (ml)'] = volume
        results['Average Intensity'] = average_intensity

    return results

def absolute_path(relative_path):
    base_path = os.path.dirname(__file__)  # Gets the directory where the script is located
    return os.path.join(base_path, relative_path)

# Eddo: RemapTransform class for inference 
class RemapLabels(MapTransform):
    def __init__(self, keys, id_map, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.id_map = id_map
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            lab = d[key]
            out = lab.clone()
            for orig, tgt in self.id_map.items():
                out[lab == orig] = tgt
            d[key] = out
        return d
    
# Eddo: SqueezeTransform to remove singleton dimension during inference
class SqueezeTransform(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            lab = d[key]
            out = lab.squeeze(0) if lab.dim() > 3 and lab.shape[0] == 1 else lab  # Remove channel dim if [1, H, W, D]
            d[key] = out
        return d

# Eddo run inference function
def run_inference(image_path, output_dir, pre_transforms, post_transforms,
                  amp_context=None, chunk_size=50, device=None, inferer=None, model=None):
    #first get image from dictionary in image_path
    data = {"image": image_path}
    # do pre_transforsm on entire image to get image in RAS orientation and as a tensor
    data = pre_transforms(data)

    # set data in test_file
    test_file = data["image"]

    #Eddo: ensure that test_file is in float when CPU is used. 
    if device.type == "cpu":
        test_file = test_file.float()

    #Eddo: If tensor is 4, then set entry as 1. 
    if test_file.ndim == 4:
        test_file = test_file.unsqueeze(0)          # [1,C,H,W,D]
    
    # set test_file to device
    test_file = test_file.to(device, non_blocking=True)
    
    # Depth is always -1 after pretransforms.  
    D = test_file.shape[-1]
    
    # Eddo: first check if CPU or GPU, if GPU then use AMP, when CPU nullcontext.  
    with amp_context, torch.inference_mode():
        # Eddo: if chunk_size is lower dan d, then normally apply the inferer. set pred.to.cpu in float16 to save RAM memory. 
        if D <= chunk_size:
            pred = inferer(test_file, model)       
            output = pred.to("cpu", dtype=torch.float16)
            del pred
        else:
            # first chunk inference from start to minimal chunk_size (default = 50). 
            start = 0
            end = min(chunk_size, D)
            first = inferer(test_file[..., start:end], model)  
            #Get shape and create equivalent tensor that we use to store the chunk in memory in cpu
            B, C, H, W, _ = first.shape
            output = torch.empty((B, C, H, W, D), dtype=torch.float16, device="cpu")
            # output first inference to cpu in float16 to save ram memory
            output[..., start:end] = first.to("cpu", dtype=torch.float16)
            #delete first to save memory
            del first
            #then continue for the rest of the chunks dependent on number of slices and save to output @ Cpu in float16. 
            for start in range(end, D, chunk_size):
                end2 = min(start + chunk_size, D)
                pred = inferer(test_file[..., start:end2], model)  
                output[..., start:end2] = pred.to("cpu", dtype=torch.float16)
                del pred

    # Eddo: drop batch dimensions from image
    single_pred = output[0]     
    # Set pred and image_metadata to postin
    post_in = {"pred": single_pred, "image": data["image"]}
    # apply post transforms
    post_out = post_transforms(post_in)
    # detach to cpu, set to numpy and as type nt.16 to reduce memory
    seg = post_out["pred"].detach().cpu().numpy().astype(np.int16)

    #get affine
    affine = data.get("image_meta_dict", {}).get("affine", None)
    # probably not needed, but if affine is none, get thet from nibabels loading. 
    if affine is None:
        try:
            affine = nib.load(image_path).affine
        except Exception:
            affine = np.eye(4)
    # define output path
    out_path = os.path.join(
        output_dir, os.path.basename(image_path).replace(".nii.gz", "_dseg.nii.gz")
    )
    # save using nibabel
    nib.save(nib.Nifti1Image(seg, affine), out_path)
    # clean up
    del test_file, output, post_in, post_out, seg
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_path
