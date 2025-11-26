import os
import logging
import sys
import json
import numpy as np
import nibabel as nib
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from monai.transforms import (MapTransform)
import gc, torch
import os, gc, torch, nibabel as nib
import shutil
from scipy import ndimage as ndi
from typing import Any, Dict, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

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

def save_nifti(data: np.ndarray, affine, header, out_path):
    new_hdr = header.copy()                            
    img = nib.Nifti1Image(data, affine, new_hdr)
    
    _, qcode = header.get_qform(coded=True)
    _, scode = header.get_sform(coded=True)
    img.set_qform(affine, int(qcode))
    img.set_sform(affine, int(scode))
    nib.save(img, out_path)

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
    affine = img.affine
    header = img.header
    
    return img, img_array,affine,header,(dim_x, dim_y, dim_z), (pixdim_x, pixdim_y, pixdim_z)

def add_slice_counts(
    results_entry: Dict[int, Dict[str, Any]],
    label_img:     np.ndarray,
    pix_dim:       Tuple[float, float, float],
    col_name:      str = "Slices with segmentation",
) -> Dict[int, Dict[str, Any]]:
    
    if label_img.ndim != 3:
        raise ValueError("label_img must be 3-D")
    
    pix_dim = tuple(float(p) for p in pix_dim)
    max_axis = int(np.argmax(pix_dim))                
    if max(pix_dim) / min(pix_dim) < 1.01:           
        max_axis = 2                                
    axes_to_reduce = tuple(ax for ax in range(3) if ax != max_axis)

    for lbl in np.unique(label_img):
        lbl = int(lbl)
        if lbl == 0:
            continue
        slice_present = np.any(label_img == lbl, axis=axes_to_reduce)  
        slice_count   = int(slice_present.sum())

        entry = results_entry.get(lbl, {"Label": lbl, "Anatomy": ""})
        entry[col_name] = slice_count
        results_entry[lbl] = entry

    return results_entry

def apply_clustering(args, mask_img, components): 
    if args.method == 'kmeans':
        clustering = KMeans(n_clusters = components, init = 'k-means++', tol = 0.001, n_init = 20, max_iter = 1000).fit(mask_img)
        labels = clustering.labels_ 
    elif args.method == 'gmm':
        clustering = GaussianMixture(n_components = components, covariance_type = 'full', init_params = 'kmeans', tol = 0.001, n_init = 20, max_iter = 1000).fit(mask_img)
        labels = clustering.predict(mask_img)
    else:
        raise ValueError("Either KMeans or GMM must be activated.")
    return labels, clustering

def calculate_thresholds(labels, mask_img, num_clusters):
    clusters = [mask_img[labels == i] for i in range(num_clusters)]
    means = [np.mean(cluster) for cluster in clusters]
    if num_clusters == 2:
        muscle_max = np.max(clusters[0]) if means[0] < means[1] else np.max(clusters[1])
        muscle_img = mask_img[mask_img <= muscle_max]
        fat_min = None # placeholder
        sorted_indices = [0, 1] if means[0] < means[1] else [1, 0]
    elif num_clusters == 3:
        sorted_clusters = sorted(zip(means, clusters, range(len(clusters))), key=lambda x: x[0])
        muscle_img = sorted_clusters[0][1]
        fat_img = sorted_clusters[2][1]
        muscle_max = np.max(muscle_img)
        fat_min= np.min(fat_img)
        sorted_indices = [x[2] for x in sorted_clusters]
    return muscle_max, fat_min, sorted_indices

def create_image_array(img_array, mask_array, label, muscle_upper, fat_lower, components):
    if components not in (2, 3):
        raise ValueError("components must be 2 or 3")

    muscle_label = (mask_array == label) 
    if components == 2:
        muscle_array    = muscle_label & (img_array <  muscle_upper)
        fat_array       = muscle_label & (img_array >= muscle_upper)
        undefined_array = np.zeros_like(img_array, dtype=bool)  # placeholder
    else:  # components == 3
        muscle_array    = muscle_label & (img_array <  muscle_upper)
        undefined_array = muscle_label & (img_array >= muscle_upper) & (img_array < fat_lower)
        fat_array       = muscle_label & (img_array >= fat_lower)
    return muscle_array, fat_array, undefined_array

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

def build_entry_dict_metrics(
    label_img: np.ndarray,
    model_config: Optional[dict],
    region: bool = False,
) -> Dict[int, Dict[str, Any]]:
    
    results_entry: Dict[int, Dict[str, Any]] = {}

    model_labels = (model_config.get("labels", []) if model_config else [])
    idx: Dict[int, str] = {}

    if region and model_labels:
        for L in model_labels:
            try:
                val = int(L.get("value"))
            except Exception:
                continue
            anatomy = str(L.get("anatomy", "")).strip()
            side    = str(L.get("side", "")).strip()
            text = f"{anatomy} {side}".strip()
            if text:
                idx[val] = text

    unmatched_labels: list[int] = []
    for lbl in np.unique(label_img):
        lbl = int(lbl)
        if lbl <= 0:
            continue

        anatomy_text = idx.get(lbl, "") if (region and idx) else ""
        if region and idx and anatomy_text == "":
            unmatched_labels.append(lbl)

        results_entry[lbl] = {
            "Anatomy": anatomy_text,
            "Label":   lbl,
        }

    if region and idx and unmatched_labels:
        logging.warning(
            "No MuscleMap anatomy-side mapping was found for the following label IDs in "
            "the current region configuration: %s. Only label numbers will be given",
            ", ".join(map(str, unmatched_labels))
        )

    return results_entry

def calculate_metrics_dixon(
    result_entry: Dict[int, Dict[str, Any]], 
    label_img: np.ndarray,
    fat_array: np.ndarray,
    water_array: np.ndarray,
    pix_dim: Tuple[float, float, float],
) -> Dict[int, Dict[str, Any]]:
    
    # raise value error when shapes do no match or when 4D is given as input
    if not (label_img.shape == water_array.shape == fat_array.shape):
        raise ValueError("label_img, water_array en fat_array moeten dezelfde shape hebben")
    if len(pix_dim) != 3:
        raise ValueError("pix_dim must be a 3-tuple (mm, mm, mm)")
    
    # fix voxel_vol_ml to calculate volume in ml
    voxel_vol_ml = (pix_dim[0] * pix_dim[1] * pix_dim[2]) / 1000.0

    # 1) Creating total fat fraction map for formula fat_signal/fat_signal + water signal. 
    denom = fat_array + water_array
    ff_map = np.divide(
        fat_array, denom,
        out=np.zeros_like(denom, dtype=np.float32),
        where=(denom != 0)
    ).astype(np.float32)

    # 2) Flatten voor aggregration and set to int64 for efficiency
    flat_ff  = ff_map.ravel()
    flat_lbl = label_img.astype(np.int64).ravel()

    # 3) get max label from image. Labels in entry but not in image will be skipped.
    max_label = int(flat_lbl.max()) if flat_lbl.size else 0

    # 4) Sum and count per label
    sum_per_lbl   = np.bincount(flat_lbl, weights=flat_ff, minlength=max_label + 1)
    count_per_lbl = np.bincount(flat_lbl, minlength=max_label + 1)

    # 5) mean per label
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_per_lbl = np.divide(
            sum_per_lbl, count_per_lbl,
            out=np.zeros_like(sum_per_lbl, dtype=np.float64),
            where=(count_per_lbl != 0)
        )
    # 6) Update result_entry with fat percentages for each label
    for _k, entry in result_entry.items():
        lbl = int(entry.get("Label", _k))  
        if lbl == 0:
            continue
        if lbl <= max_label and count_per_lbl[lbl] > 0:
            fat_pct = round(float(mean_per_lbl[lbl] * 100.0), 2)
            vol_ml  = round(float(count_per_lbl[lbl] * voxel_vol_ml), 2)
        else:
            fat_pct = np.nan
            vol_ml  = np.nan
        entry.update({
            "Fat (%)":     fat_pct,
            "Volume (ml)": vol_ml,
        })
    # 7) Return updated result_entry dictionary
    return result_entry

def calculate_metrics_average(
    result_entry: Dict[int, Dict[str, Any]],
    label_img: np.ndarray,
    img_array: np.ndarray,
    pix_dim: Tuple[float, float, float],
) -> Dict[int, Dict[str, Any]]:
    
    #Raise ValueError when mismatch or image not in 3D
    if label_img.shape != img_array.shape:
        raise ValueError("label_img and img_array must have the same shape")
    if len(pix_dim) != 3:
        raise ValueError("pix_dim must be a 3-tuple (mm, mm, mm)")
    
    # fix voxel_vol_ml to calculate volume in ml 
    voxel_vol_ml = (pix_dim[0] * pix_dim[1] * pix_dim[2]) / 1000.0

    # Vectorized aggregations
    flat_lbl = label_img.astype(np.int64).ravel()
    flat_val = img_array.astype(np.float64).ravel()
    max_label = int(flat_lbl.max()) if flat_lbl.size else 0
    
    # Vectorized calculations for sum and count
    sum_per_lbl   = np.bincount(flat_lbl, weights=flat_val, minlength=max_label + 1)
    count_per_lbl = np.bincount(flat_lbl, minlength=max_label + 1)

    # ignore dividing by zero error and only divide where count_per_lbl is > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_per_lbl = np.divide(
            sum_per_lbl, count_per_lbl,
            out=np.zeros_like(sum_per_lbl, dtype=np.float64),
            where=(count_per_lbl != 0)
        )
    labels_present = np.flatnonzero(count_per_lbl)
    for lbl in labels_present:
        if lbl == 0:
            continue
        avg_int = round(float(mean_per_lbl[lbl]), 2)
        vol_ml  = round(float(count_per_lbl[lbl] * voxel_vol_ml), 2)

        entry = result_entry.get(int(lbl), {"Anatomy": "", "Label": int(lbl)})
        entry.update({
            "Average Intensity": avg_int,
            "Volume (ml)":       vol_ml,
        })
        result_entry[int(lbl)] = entry

    for _k, entry in list(result_entry.items()):
        lbl = int(entry.get("Label", _k))  
        if lbl == 0:
            continue
        if lbl > max_label or count_per_lbl[lbl] == 0:
            entry.setdefault("Average Intensity", np.nan)
            entry.setdefault("Volume (ml)",       np.nan)

    return result_entry

def calculate_metrics_thresholding(
    args,
    results_entry: Dict[str, Any],                            
    label_img: np.ndarray,                
    img_array: np.ndarray,                 
    affine: np.ndarray,
    header:  np.ndarray,                   
    pix_dim: Tuple[float, float, float],  
    components: int,                      
    output_dir: Union[str, Path],          
    id_part: str = "",                   
) -> Dict[str, Any]:
    
    # raise value errors if components is not 2/3 or when mismatch in shape
    if components not in (2, 3):
        raise ValueError("components must be 2 or 3")
    if label_img.shape != img_array.shape:
        raise ValueError("label_img and img_array must have the same shape")
    
    #prepare empty image array to build up the fat, muscle (and in 3 component; undefined)maps    
    total_muscle_image    = np.zeros_like(img_array, dtype=bool)
    total_fat_image       = np.zeros_like(img_array, dtype=bool)
    total_undefined_image = np.zeros_like(img_array, dtype=bool)
    combined_mask = np.zeros_like(label_img, dtype=np.uint8)  
    
    # if GMM is chosen, we will also create an empty array in float32 for each component to store softprob. 
    if args.method == 'gmm':
        total_probability_maps = [np.zeros(label_img.shape, dtype=np.float32)
                                for _ in range(components)]
        
    # determine voxel vol ml to easily calculate volume from pixdim  
    voxel_vol_ml = (pix_dim[0] * pix_dim[1] * pix_dim[2]) / 1000.0  

    # build up the dictionary for 2 or 3 clusters and apply
    for lbl in np.unique(label_img):
        # we will skip background (label == 0 in each model)
        if lbl == 0:
            continue

        #create mask specific the voxels from foreground label
        mask = (label_img == lbl)

        #reshape mask so that it can be used for thresholding (1D)
        mask_img = img_array[mask].reshape(-1, 1)

        # apply the clustering function and we get two (or three) maps with voxels
        labels, clustering = apply_clustering(args,
            mask_img, components
        )

        # calculate thresholds from clustering 
        muscle_max, fat_min, sorted_indices = calculate_thresholds(labels, mask_img, components)
        
        # determine the number of voxels over the 1D vector
        N = mask_img.size

        # determine the total_volume for the label
        total_volume = N * voxel_vol_ml

        # use thresholds (muscle max for bimodal and muscle_max + fat_min for trimodal) to build up image and calculate percentage
        if components == 2:
            # iteravilely build up boolean fat and muscle maps
            muscle_array, fat_array, _ = create_image_array(img_array, label_img, lbl, muscle_max, fat_min, components)
            total_muscle_image |= muscle_array
            total_fat_image |= fat_array
            combined_mask[muscle_array] = 1
            combined_mask[fat_array]    = 4

            # fat and muscle calculations
            muscle_percentage = 100.0 * np.mean((mask_img.ravel() <= muscle_max))
            fat_percentage = 100 - muscle_percentage
            
            # volume calculations
            muscle_voxels = np.count_nonzero(mask_img <= muscle_max)
            muscle_volume = muscle_voxels * voxel_vol_ml
            fat_volume = (N - muscle_voxels) * voxel_vol_ml 

        if components == 3: 
            # iteraively build up fat and muscle maps
            muscle_array, fat_array, undefined_array = create_image_array(img_array, label_img, lbl, muscle_max, fat_min, components)
            total_muscle_image |= muscle_array
            total_fat_image |= fat_array  
            total_undefined_image |= undefined_array

            combined_mask[muscle_array]    = 1   
            combined_mask[undefined_array] = 7 
            combined_mask[fat_array]       = 4  

            #fat,muscle and undefined calculations
            muscle_percentage    = np.nan if N == 0 else 100.0 * np.mean(mask_img <  muscle_max)
            undefined_percentage = np.nan if N == 0 else 100.0 * np.mean((mask_img >= muscle_max) & (mask_img < fat_min))
            fat_percentage       = np.nan if N == 0 else 100.0 * np.mean(mask_img >= fat_min)

            # volume calculations
            muscle_voxels = np.count_nonzero(mask_img <= muscle_max)
            muscle_volume = muscle_voxels * voxel_vol_ml
            undefined_voxels = np.count_nonzero((mask_img > muscle_max) & (mask_img < fat_min))
            undefined_volume = undefined_voxels * voxel_vol_ml
            fat_voxels = np.count_nonzero(mask_img >= fat_min)
            fat_volume = fat_voxels * voxel_vol_ml

        entry = results_entry.get(int(lbl), {"Anatomy": "", "Label": int(lbl)})
        entry.update({
            "Muscle (%)":         (np.nan if muscle_percentage is None else round(float(muscle_percentage), 2)),
            "Fat (%)":            (np.nan if fat_percentage is None else round(float(fat_percentage), 2)),
            "Total volume (ml)":  (np.nan if total_volume is None else round(float(total_volume), 2)),
            "Fat volume (ml)":    (np.nan if fat_volume is None else round(float(fat_volume), 2)),
            "Muscle volume (ml)": (np.nan if muscle_volume is None else round(float(muscle_volume), 2)),
        })

        if components == 3:
            entry["Undefined (%)"]         = (np.nan if undefined_percentage is None else round(float(undefined_percentage), 2))
            entry["Undefined volume (ml)"] = (np.nan if undefined_volume     is None else round(float(undefined_volume),     2))

        results_entry[int(lbl)] = entry

        if args.method == 'gmm':
            probability_maps = clustering.predict_proba(mask_img)             
            sorted_probability_maps = probability_maps[:, sorted_indices]      
            for comp_idx in range(components):
                 total_probability_maps[comp_idx][mask] += sorted_probability_maps[:, comp_idx].astype(np.float32)

    save_nifti(total_muscle_image.astype(np.uint8), affine,header, os.path.join(output_dir, f'{id_part}_{args.method}_{args.components}component_muscle_seg.nii.gz'))
    save_nifti(combined_mask, affine, header,
           os.path.join(output_dir,
                        f"{id_part}_{args.method}_{components}component_combined_seg.nii.gz"))
    if components == 3:
        save_nifti(total_undefined_image.astype(np.uint8), affine,header, os.path.join(output_dir, f'{id_part}_{args.method}_{args.components}component_undefined_seg.nii.gz'))
    save_nifti(total_fat_image.astype(np.uint8), affine,header, os.path.join(output_dir, f'{id_part}_{args.method}_{args.components}component_fat_seg.nii.gz'))
    
    if args.method == 'gmm':
        if components == 3:
            component_names = ["muscle", "undefined", "fat"]
        else:
            component_names = ["muscle", "fat"]
        for comp_idx, comp_name in enumerate(component_names):
            out_path = os.path.join(
                output_dir,
                f"{id_part}_gmm_{comp_name}_{components}component_softseg.nii.gz"
            )
        save_nifti(total_probability_maps[comp_idx], affine, header, out_path)
    return results_entry

def results_entry_to_dataframe(results_entry: dict[int, dict]) -> pd.DataFrame:
    rows = []
    for lbl, entry in results_entry.items():
        label_val = int(entry.get("Label", lbl))
        row = {"Label": label_val}
        row.update(entry)
        rows.append(row)
    df = pd.DataFrame(rows)
    if "Label" in df.columns:
        df = df.drop_duplicates(subset=["Label"]).sort_values("Label")   
    return df

def absolute_path(relative_path):
    base_path = os.path.dirname(__file__)  # Gets the directory where the script is located
    return os.path.join(base_path, relative_path)

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
    
def connected_chunks(
    seg: np.ndarray,
    labels: Optional[np.ndarray] = None,
    connectivity: int = 3,  # 3=26-connectivity
) -> np.ndarray:
    
    """
    Keeps only the largest connected component per label in a multi-label segmentation.
    - Supports both 3D (X,Y,Z) and 4D (1,X,Y,Z) arrays.
    - Incorporated to get optimize RAM memory management during inference for large images
    - 
    """
    #Ensure that is nparray
    seg = np.asarray(seg)

    # Add channel dimension if necessary (to unify shape to 4D)
    remove_dim = False
    if seg.ndim == 3:
        seg_ch = seg[None, ...]
        remove_dim = True
    elif seg.ndim == 4 and seg.shape[0] == 1:
        seg_ch = seg
    else:
        raise ValueError(f"Expected (X,Y,Z) or (1,X,Y,Z), got {seg.shape}")

    # find labels excluding background (0)
    if labels is None:
        labels = np.unique(seg_ch)
    labels = labels[labels != 0]
    if labels.size == 0:
        result = seg_ch.astype(np.int16, copy=False)
        return result[0] if remove_dim else result

    # Extract the 3D volume from channel 0 for processing
    vol = seg_ch[0]

    # Connectivity structure for 3D, rank 3 = 26-connectivity to be not to conversative
    structure = ndi.generate_binary_structure(rank=3, connectivity=connectivity)

    # Buffers for 3D mask and labels
    mask3d = np.empty(vol.shape, dtype=bool)
    lab3d  = np.empty(vol.shape, dtype=np.int32)

    # Process each label independently on the 3D volume
    for lab_id in labels:
        np.equal(vol, lab_id, out=mask3d)
        if not mask3d.any():
            continue
        # Label connected components on mask3d
        ndi.label(mask3d, structure=structure, output=lab3d)
        max_lab = int(lab3d.max())
        if max_lab <= 1:
            continue

        # Compute sizes and pick the largest component
        counts = np.bincount(lab3d.ravel())
        keep = counts[1:].argmax() + 1
        del counts

        # Zero out everything except the largest component for this label
        np.logical_and(mask3d, lab3d != keep, out=mask3d)
        vol[mask3d] = 0

    # Write back the processed 3D volume into output array
    seg_ch[0] = vol

    # Cleanup
    del mask3d, lab3d

    # Convert to int16 and drop channel dim if needed
    result = seg_ch.astype(np.int16, copy=False)
    if remove_dim:
        result = result[0]
    return result

def is_nifti(path: str) -> bool:
    p = path.lower()
    return p.endswith(".nii.gz") or p.endswith(".nii")

def _get_memory_usage():
    """Get current CPU and GPU memory usage in GB"""
    import psutil
    
    # CPU memory - process usage and total system RAM
    process = psutil.Process()
    cpu_used = process.memory_info().rss / 1024**3
    cpu_total = psutil.virtual_memory().total / 1024**3
    
    # GPU memory - allocated, reserved, and total
    gpu_used = 0
    gpu_reserved = 0
    gpu_total = 0
    if torch.cuda.is_available():
        gpu_used = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    return cpu_used, cpu_total, gpu_used, gpu_reserved, gpu_total


def get_peak_memory(reset: bool = False):
    """Return peak CPU and GPU memory usage (GB).

    Args:
        reset: if True and CUDA is available, reset CUDA peak stats after reading.

    Returns:
        Tuple (peak_cpu_gb, peak_gpu_allocated_gb, peak_gpu_reserved_gb)
    """
    import psutil
    peak_cpu_gb = 0.0
    peak_gpu_allocated_gb = 0.0
    peak_gpu_reserved_gb = 0.0

    try:
        process = psutil.Process()
        # Use current RSS as a conservative peak indicator (portable)
        peak_cpu_gb = process.memory_info().rss / 1024**3
    except Exception:
        peak_cpu_gb = 0.0

    if torch.cuda.is_available():
        try:
            peak_gpu_allocated_gb = torch.cuda.max_memory_allocated() / 1024**3
        except Exception:
            peak_gpu_allocated_gb = 0.0
        try:
            peak_gpu_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
        except Exception:
            peak_gpu_reserved_gb = 0.0

        if reset:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    return peak_cpu_gb, peak_gpu_allocated_gb, peak_gpu_reserved_gb

def _plot_performance_metrics(metrics_data, output_path):
    """Create a line plot of CPU and GPU usage across processing stages"""
    if not metrics_data:
        return
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Extract data
    x_positions = list(range(len(metrics_data)))
    gpu_used = [m['gpu_used'] for m in metrics_data]
    gpu_reserved = [m['gpu_reserved'] for m in metrics_data]
    gpu_total = [m['gpu_total'] for m in metrics_data]
    cpu_used = [m['cpu_used'] for m in metrics_data]
    cpu_total = [m['cpu_total'] for m in metrics_data]
    stages = [m['stage'] for m in metrics_data]
    chunk_labels = [m['label'] for m in metrics_data]
    
    # Calculate cumulative sums (only increasing)
    cpu_cumulative = []
    gpu_cumulative = []
    cpu_sum = 0
    gpu_sum = 0
    for cpu, gpu in zip(cpu_used, gpu_used):
        cpu_sum += cpu
        gpu_sum += gpu
        cpu_cumulative.append(cpu_sum)
        gpu_cumulative.append(gpu_sum)
    
    # Get total memory values (should be constant)
    gpu_total_val = gpu_total[0] if gpu_total and gpu_total[0] > 0 else None
    cpu_total_val = cpu_total[0] if cpu_total else None
    
    # Plot GPU on left y-axis
    color_gpu = '#2563eb'  # Blue
    color_gpu_reserved = '#7c3aed'  # Purple
    color_gpu_cumulative = '#06b6d4'  # Cyan
    color_gpu_total = '#93c5fd'  # Light blue
    ax1.set_xlabel('Processing Stage', fontsize=12, fontweight='bold')
    ax1.set_ylabel('GPU Memory (GB)', color=color_gpu, fontsize=12, fontweight='bold')
    
    # Plot GPU used (allocated)
    ax1.plot(x_positions, gpu_used, color=color_gpu, linewidth=2, 
             marker='o', markersize=4, label='GPU Used (Allocated)')
    
    # Plot GPU reserved
    ax1.plot(x_positions, gpu_reserved, color=color_gpu_reserved, linewidth=2, 
             marker='^', markersize=4, linestyle='--', label='GPU Reserved')
    
    # Plot GPU cumulative
    ax1.plot(x_positions, gpu_cumulative, color=color_gpu_cumulative, linewidth=2.5, 
             marker='*', markersize=5, linestyle=':', alpha=0.7, label='GPU Cumulative')
    
    # Plot GPU total as horizontal line if available
    if gpu_total_val:
        ax1.axhline(y=gpu_total_val, color=color_gpu_total, linestyle='--', 
                   linewidth=1.5, alpha=0.7, label=f'GPU Total ({gpu_total_val:.1f} GB)')
    
    ax1.tick_params(axis='y', labelcolor=color_gpu)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Create right y-axis for CPU
    ax2 = ax1.twinx()
    color_cpu = '#dc2626'  # Red
    color_cpu_cumulative = '#f59e0b'  # Orange
    color_cpu_total = '#fca5a5'  # Light red
    ax2.set_ylabel('CPU Memory (GB)', color=color_cpu, fontsize=12, fontweight='bold')
    
    # Plot CPU used
    ax2.plot(x_positions, cpu_used, color=color_cpu, linewidth=2,
             marker='s', markersize=4, label='CPU Used')
    
    # Plot CPU cumulative
    ax2.plot(x_positions, cpu_cumulative, color=color_cpu_cumulative, linewidth=2.5,
             marker='*', markersize=5, linestyle=':', alpha=0.7, label='CPU Cumulative')
    
    # Plot CPU total as horizontal line if available
    if cpu_total_val:
        ax2.axhline(y=cpu_total_val, color=color_cpu_total, linestyle='--', 
                   linewidth=1.5, alpha=0.7, label=f'CPU Total ({cpu_total_val:.1f} GB)')
    
    ax2.tick_params(axis='y', labelcolor=color_cpu)
    
    # Set x-axis ticks and labels - show only every 3rd label (since pattern repeats: preprocess, predict, postprocess)
    # Find stage boundaries for bold labels
    stage_starts = {}
    for i, stage in enumerate(stages):
        if stage not in stage_starts:
            stage_starts[stage] = i
    
    # Set all tick positions but only label every 3rd one
    ax1.set_xticks(x_positions)
    x_labels = []
    for i, label in enumerate(chunk_labels):
        if i % 3 == 0:  # Show every 3rd label
            x_labels.append(label)
        else:
            x_labels.append('')  # Empty string for unlabeled ticks
    ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=6)
    
    # Add bold stage markers at the top
    for stage, pos in stage_starts.items():
        ax1.axvline(x=pos, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
        ax1.text(pos, ax1.get_ylim()[1] * 1.02, stage.upper(), 
                fontweight='bold', fontsize=11, ha='left')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    # Minimal theme adjustments
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    plt.title('CPU and GPU Memory Usage During Processing (Used vs Total)', 
             fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Performance plot saved to: {output_path}")

def _make_out_path(image_path, output_dir, tag="_dseg"):
    fname = os.path.basename(image_path)
    if fname.endswith(".nii.gz"):
        base = fname[:-7]
    elif fname.endswith(".nii"):
        base = fname[:-4]
    return os.path.join(output_dir, f"{base}{tag}.nii.gz")

def run_inference(
    image_path,
    output_dir,
    pre_transforms,
    post_transforms,
    amp_context=None,
    chunk_size=50,
    device=None,
    inferer=None,
    model=None,
):
    # Initialize performance tracking
    metrics_data = []
    
    # 1) Load header + data
    out_path = _make_out_path(image_path, output_dir, "_dseg")
    img_nii  = nib.load(image_path)
    affine   = img_nii.affine.copy()
    header   = img_nii.header.copy()
    img_data = img_nii.get_fdata().astype(np.float32)
    D        = img_data.shape[-1]
    
    if D <= chunk_size:
        # Track preprocessing
        logging.info("="*80)
        logging.info("STARTING PREPROCESSING - Chunk 1")
        logging.info("="*80)
        CheckTransformMemory.reset_tracking()
        
        data   = {"image": image_path}
        data   = pre_transforms(data)
        cpu_used, cpu_total, gpu_used, gpu_reserved, gpu_total = _get_memory_usage()
        metrics_data.append({
            'stage': 'preprocessing',
            'label': 'Chunk 1',
            'cpu_used': cpu_used,
            'cpu_total': cpu_total,
            'gpu_used': gpu_used,
            'gpu_reserved': gpu_reserved,
            'gpu_total': gpu_total
        })
        
        tensor = data["image"]
        if device.type == "cpu":
            tensor = tensor.float()
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(0)              
        tensor = tensor.to(device, non_blocking=True)

        # Track prediction
        logging.info("="*80)
        logging.info("STARTING PREDICTION - Chunk 1")
        logging.info("="*80)
        cpu_used, cpu_total, gpu_used, gpu_reserved, gpu_total = _get_memory_usage()
        logging.info(f"Pre-prediction memory - RAM: {cpu_used:.2f}/{cpu_total:.1f} GB | GPU VRAM: {gpu_used:.2f} (Reserved: {gpu_reserved:.2f})/{gpu_total:.1f} GB")
        with amp_context, torch.inference_mode():
            pred = inferer(tensor, model)
        cpu_used, cpu_total, gpu_used, gpu_reserved, gpu_total = _get_memory_usage()
        logging.info(f"Post-prediction memory - RAM: {cpu_used:.2f}/{cpu_total:.1f} GB | GPU VRAM: {gpu_used:.2f} (Reserved: {gpu_reserved:.2f})/{gpu_total:.1f} GB")
        metrics_data.append({
            'stage': 'prediction',
            'label': 'Chunk 1',
            'cpu_used': cpu_used,
            'cpu_total': cpu_total,
            'gpu_used': gpu_used,
            'gpu_reserved': gpu_reserved,
            'gpu_total': gpu_total
        })

        single_pred = pred.squeeze(0).squeeze(0)     
        post_in = {
            "pred": single_pred,
            "image": data["image"],
            "image_meta_dict": data["image_meta_dict"],
        }
        del data
        
        # Track postprocessing
        logging.info("="*80)
        logging.info("STARTING POSTPROCESSING - Chunk 1")
        logging.info("="*80)
        post_out    = post_transforms(post_in)
        cpu_used, cpu_total, gpu_used, gpu_reserved, gpu_total = _get_memory_usage()
        metrics_data.append({
            'stage': 'postprocessing',
            'label': 'Chunk 1',
            'cpu_used': cpu_used,
            'cpu_total': cpu_total,
            'gpu_used': gpu_used,
            'gpu_reserved': gpu_reserved,
            'gpu_total': gpu_total
        })
        
        seg_tensor  = post_out["pred"].detach().cpu().to(torch.int16)
        seg_np      = seg_tensor.numpy()
        full_seg = connected_chunks(seg_np)
        nib.save(nib.Nifti1Image(full_seg, affine, header), out_path)
        del seg_np  
        # cleanup
        del tensor, pred, single_pred, post_in, post_out, seg_tensor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Generate performance plot
        plot_path = out_path.replace('_dseg.nii.gz', '_performance.png')
        _plot_performance_metrics(metrics_data, plot_path)
        
        return out_path

    temp_dir = os.path.join(output_dir, "temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)
    gc.collect()  
    chunk_files = []
    for start in range(0, D, chunk_size):
        end       = min(start + chunk_size, D)
        vol_chunk = img_data[..., start:end]
        chunk_path = os.path.join(temp_dir, f"chunk_{start}_{end}.nii.gz")
        nib.save(nib.Nifti1Image(vol_chunk, affine, header), chunk_path)
        del vol_chunk  
        chunk_files.append({"image": chunk_path, "start": start, "end": end})

    del img_data, img_nii
    gc.collect()

    for idx, entry in enumerate(chunk_files, 1):
        chunk_label = f"Chunk {idx}"
        
        # Track preprocessing
        logging.info("="*80)
        logging.info(f"STARTING PREPROCESSING - {chunk_label}")
        logging.info("="*80)
        CheckTransformMemory.reset_tracking()
        
        data   = {"image": entry["image"]}
        data   = pre_transforms(data)
        cpu_used, cpu_total, gpu_used, gpu_reserved, gpu_total = _get_memory_usage()
        metrics_data.append({
            'stage': 'preprocessing',
            'label': chunk_label,
            'cpu_used': cpu_used,
            'cpu_total': cpu_total,
            'gpu_used': gpu_used,
            'gpu_reserved': gpu_reserved,
            'gpu_total': gpu_total
        })
        
        tensor = data["image"]
        if device.type == "cpu":
            tensor = tensor.float()
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(device, non_blocking=True)

        # Track prediction
        logging.info("="*80)
        logging.info(f"STARTING PREDICTION - {chunk_label}")
        logging.info("="*80)
        cpu_used, cpu_total, gpu_used, gpu_reserved, gpu_total = _get_memory_usage()
        logging.info(f"Pre-prediction memory - RAM: {cpu_used:.2f}/{cpu_total:.1f} GB | GPU VRAM: {gpu_used:.2f} (Reserved: {gpu_reserved:.2f})/{gpu_total:.1f} GB")
        with amp_context, torch.inference_mode():
            pred = inferer(tensor, model)
        cpu_used, cpu_total, gpu_used, gpu_reserved, gpu_total = _get_memory_usage()
        logging.info(f"Post-prediction memory - RAM: {cpu_used:.2f}/{cpu_total:.1f} GB | GPU VRAM: {gpu_used:.2f} (Reserved: {gpu_reserved:.2f})/{gpu_total:.1f} GB")
        metrics_data.append({
            'stage': 'prediction',
            'label': chunk_label,
            'cpu_used': cpu_used,
            'cpu_total': cpu_total,
            'gpu_used': gpu_used,
            'gpu_reserved': gpu_reserved,
            'gpu_total': gpu_total
        })

        single_pred = pred.squeeze(0).squeeze(0)
        post_in = {
            "pred": single_pred,
            "image": data["image"],
            "image_meta_dict": data["image_meta_dict"],
        }
        del data  
        
        # Track postprocessing
        logging.info("="*80)
        logging.info(f"STARTING POSTPROCESSING - {chunk_label}")
        logging.info("="*80)
        post_out = post_transforms(post_in)
        cpu_used, cpu_total, gpu_used, gpu_reserved, gpu_total = _get_memory_usage()
        metrics_data.append({
            'stage': 'postprocessing',
            'label': chunk_label,
            'cpu_used': cpu_used,
            'cpu_total': cpu_total,
            'gpu_used': gpu_used,
            'gpu_reserved': gpu_reserved,
            'gpu_total': gpu_total
        })
        
        seg_tensor = post_out["pred"].detach().cpu().to(torch.int16)
        seg_np = seg_tensor.numpy()
        seg_path = os.path.join(
            temp_dir,
            f"seg_{entry['start']}_{entry['end']}.nii.gz"
        )
        nib.save(nib.Nifti1Image(seg_np, affine, header), seg_path)
        entry["seg"] = seg_path
        del seg_np  

        del tensor, pred, single_pred, post_in, post_out, seg_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()  

    dims     = header.get_data_shape()  
    full_seg = np.zeros(dims, dtype=np.int16)
    for entry in chunk_files:
        s, e, sp = entry["start"], entry["end"], entry["seg"]
        vol_seg  = nib.load(sp).get_fdata().astype(np.int16)
        full_seg[..., s:e] = vol_seg
        gc.collect()
        del vol_seg 

    gc.collect()  
    full_seg = connected_chunks(full_seg)
    nib.save(nib.Nifti1Image(full_seg, affine, header), out_path)

    # final cleanup
    del full_seg
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Generate performance plot
    plot_path = out_path.replace('_dseg.nii.gz', '_performance.png')
    _plot_performance_metrics(metrics_data, plot_path)
    
    return out_path

class CheckTransformMemory(MapTransform):
    """Check CPU and GPU memory usage before/after each transform"""
    # Class variables to track memory across transform calls
    _last_cpu = None
    _last_gpu = None
    _peak_cpu = 0
    _peak_gpu = 0
    
    def __init__(self, transform_name, keys=["image"], allow_missing_keys=True):
        super().__init__(keys if keys else [], allow_missing_keys)
        self.transform_name = transform_name
    
    @classmethod
    def reset_tracking(cls):
        """Reset memory tracking for a new chunk"""
        cls._last_cpu = None
        cls._last_gpu = None
        cls._peak_cpu = 0
        cls._peak_gpu = 0
        
    def __call__(self, data):
        import psutil
        
        # Record current memory
        process = psutil.Process()
        cpu_current = process.memory_info().rss / 1024**3
        
        gpu_current = 0
        if torch.cuda.is_available():
            gpu_current = torch.cuda.memory_allocated() / 1024**3
        
        # Calculate delta from last measurement
        if CheckTransformMemory._last_cpu is None:
            cpu_delta = 0
            gpu_delta = 0
            CheckTransformMemory._peak_cpu = cpu_current
            CheckTransformMemory._peak_gpu = gpu_current
        else:
            cpu_delta = cpu_current - CheckTransformMemory._last_cpu
            gpu_delta = gpu_current - CheckTransformMemory._last_gpu
            
            # Track peaks
            if cpu_current > CheckTransformMemory._peak_cpu:
                CheckTransformMemory._peak_cpu = cpu_current
            if gpu_current > CheckTransformMemory._peak_gpu:
                CheckTransformMemory._peak_gpu = gpu_current
        
        # Update last values for next call
        CheckTransformMemory._last_cpu = cpu_current
        CheckTransformMemory._last_gpu = gpu_current
        
        # Determine spike/dip indicators with color codes
        # ANSI color codes: green for spike, red for dip
        GREEN = '\033[92m'
        RED = '\033[91m'
        RESET = '\033[0m'
        
        cpu_indicator = ""
        gpu_indicator = ""
        
        if cpu_delta > 0.1:  # Spike threshold: 100MB
            cpu_indicator = f" {GREEN}↑ SPIKE{RESET}"
        elif cpu_delta < -0.1:  # Dip threshold: -100MB
            cpu_indicator = f" {RED}↓ DIP{RESET}"
        
        if gpu_delta > 0.1:
            gpu_indicator = f" {GREEN}↑ SPIKE{RESET}"
        elif gpu_delta < -0.1:
            gpu_indicator = f" {RED}↓ DIP{RESET}"
        
        logging.info(f"[{self.transform_name}] RAM: {cpu_current:.2f} GB (Δ {cpu_delta:+.3f} GB){cpu_indicator} | GPU VRAM: {gpu_current:.2f} GB (Δ {gpu_delta:+.3f} GB){gpu_indicator}")
        
        # Pass through data unchanged
        return dict(data)