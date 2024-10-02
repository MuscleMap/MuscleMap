#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# For usage, type: python mm_segment_performance_metrics.py 

Author: Brian Kim
"""

#%% Import packages
import logging
import warnings
import numpy as np
import nibabel as nib
import re
import os
from glob import glob
import pandas as pd
from scipy import ndimage
from MetricsReloaded.utility.utils import MorphologyOps
import sys
print("Command line arguments received:", sys.argv)

def get_parser():
    parser = argparse.ArgumentParser(
        description="Calculate common image segmentation performance metrics.")
    
    # Required arguments
    required = parser.add_argument_group("Required")
    
    required.add_argument("-i", '--input_dir', required=True, type=str,
                          help="Input directory which has three folders: images, mm and gt.")
    
    # Optional arguments
    optional = parser.add_argument_group("Optional")
    required.add_argument("-o", '--output_dir', required=False, type=str,
                            help="Output directory to save the results, output file name suffix = dseg. If left empty, saves to current working directory.")
    
    optional.add_argument("-g", '--use_GPU', required=False, default = 'Y', type=str ,choices=['Y', 'N'],
                        help="If N will use the cpu even if a cuda enabled device is identified. Default is Y.")
    return parser

#%% Initialize - ADJUST FOR REGION
tissue_labels = [
    ('Supraspinatus', 0),
    ('Subscapularis', 1),
    ('Infraspinatus', 2),
    ('TeresMinor', 3),
    ('Deltoid', 4),
    ('TeresMajor', 5),
    ('Scapula', 6),
    ('Clavicle', 7)
    ]

all_subject_data = []
results = pd.DataFrame()

#%% Define basic overlap functions
def fp_map(ref, pred):
    ref_float = np.asarray(ref, dtype=np.float32)
    pred_float = np.asarray(pred, dtype=np.float32)
    return np.asarray((pred_float - ref_float) > 0.0, dtype=np.float32)

def fn_map(ref, pred):
    ref_float = np.asarray(ref, dtype=np.float32)
    pred_float = np.asarray(pred, dtype=np.float32)
    return np.asarray((ref_float - pred_float) > 0.0, dtype=np.float32)

def tp_map(ref, pred):
    ref_float = np.asarray(ref, dtype=np.float32)
    pred_float = np.asarray(pred, dtype=np.float32)
    return np.asarray((ref_float + pred_float) > 1.0, dtype=np.float32)

def tn_map(ref, pred):
    ref_float = np.asarray(ref, dtype=np.float32)
    pred_float = np.asarray(pred, dtype=np.float32)
    return np.asarray((ref_float + pred_float) < 0.5, dtype=np.float32)

def union_map(ref, pred):
    return np.asarray((ref + pred) > 0.5, dtype=np.float32)

def intersection_map(ref, pred):
    return np.multiply(ref, pred)

def n_pos_ref(ref):
    return np.sum(ref)

def n_neg_ref(ref):
    return np.sum(1 - ref)

def n_pos_pred(pred):
    """
    Returns the number of positive elements in the prediction
    """
    return np.sum(pred)

def n_neg_pred(pred):
    """
    Returns the number of negative elements in the prediction
    """
    return np.sum(1 - pred)

def fp(ref, pred):
    """
    Calculates the number of FP as sum of elements in FP_map
    """
    return np.sum(fp_map(ref, pred))

def fn(ref, pred):
    """
    Calculates the number of FN as sum of elements of FN_map
    """
    return np.sum(fn_map(ref, pred))

def tp(ref, pred):
    """
    Returns the number of true positive (TP) elements
    """
    return np.sum(tp_map(ref, pred))

def tn(ref, pred):
    """
    Returns the number of True Negative (TN) elements
    """
    return np.sum(tn_map(ref, pred))

def n_intersection(ref, pred):
    """
    Returns the number of elements in the intersection of reference and prediction (=TP)
    """
    return np.sum(intersection_map(ref, pred))

def n_union(ref, pred):
    """
    Returns the number of elements in the union of reference and prediction

    .. math::

        U = {\vert} Pred {\vert} + {\vert} Ref {\vert} - TP

    """
    return np.sum(union_map(ref, pred))

def border_distance(seg_class, gt_class, connectivity_type=1, pixdim=None):
    """
    This function determines the map of distance from the borders of the
    prediction and the reference and the border maps themselves

    :return: distance_border_ref, distance_border_pred, border_ref, border_pred
    """
    border_ref = MorphologyOps(gt_class, connectivity_type).border_map()
    border_pred = MorphologyOps(seg_class, connectivity_type).border_map()
    distance_ref = ndimage.distance_transform_edt(
        1 - border_ref, sampling=pixdim
    )
    distance_pred = ndimage.distance_transform_edt(
        1 - border_pred, sampling=pixdim
    )
    distance_border_pred = border_ref * distance_pred
    distance_border_ref = border_pred * distance_ref
    return distance_border_ref, distance_border_pred, border_ref, border_pred

#%% Define performance metric functions
def dsc(seg_class, gt_class):

    numerator = 2 * np.sum(tp_map(gt_class, seg_class))
    denominator = np.sum(n_pos_pred(seg_class)) + np.sum(n_pos_ref(gt_class))
    if denominator == 0:
        warnings.warn("Both Prediction and Reference are empty - set to 1 as correct solution even if not defined")
        return 1
    else:
        return numerator / denominator
    
def normalised_surface_distance(seg_class, gt_class, dict_args={}):
    """
    Calculates the normalised surface distance (NSD) between prediction and reference
    using the distance parameter :math:`{\\tau}`

    Stanislav Nikolov, Sam Blackwell, Alexei Zverovitch, Ruheena Mendes, Michelle Livne, Jeffrey De Fauw, Yojan Patel,
    Clemens Meyer, Harry Askham, Bernadino Romera-Paredes, et al. 2021. Clinically applicable segmentation of head
    and neck anatomy for radiotherapy: deep learning algorithm development and validation study. Journal of Medical
    Internet Research 23, 7 (2021), e26151.

    .. math::

        NSD(A,B)^{(\\tau)} = \dfrac{|S_{A} \cap Bord_{B,\\tau}| + |S_{B} \cup Bord_{A,\\tau}|}{|S_{A}| + S_{B}}

    :return: NSD
    """
    if "nsd" in dict_args.keys():
        tau = dict_args["nsd"]
    else:
        tau = tau_value
    dist_ref, dist_pred, border_ref, border_pred = border_distance(seg_class, gt_class)
    reg_ref = np.where(
        dist_ref <= tau, np.ones_like(dist_ref), np.zeros_like(dist_ref)
    )
    reg_pred = np.where(
        dist_pred <= tau, np.ones_like(dist_pred), np.zeros_like(dist_pred)
    )
    numerator = np.sum(border_pred * reg_ref) + np.sum(border_ref * reg_pred)
    denominator = np.sum(border_ref) + np.sum(border_pred)
    return numerator / denominator

def absolute_volume_difference(seg_class, gt_class, water_img): 
    dim_x, dim_y, dim_z = water_img.header['dim'][1:4]
    pixdim_x, pixdim_y, pixdim_z = water_img.header['pixdim'][1:4]
    
    
    seg_volume = np.around(seg_class.sum() * pixdim_x * pixdim_y * pixdim_z / 1000, decimals=2)
    gt_volume = np.around(gt_class.sum() * pixdim_x * pixdim_y * pixdim_z / 1000, decimals=2)
    
    avb = abs(seg_volume - gt_volume)
  
    return avb

def intersection_over_union(seg_class, gt_class):
    if np.sum(seg_class) == 0 and np.sum(gt_class) == 0:
        warnings.warn("Both reference and prediction are empty")
        return np.nan
    return np.sum(intersection_map(gt_class, seg_class)) / np.sum(union_map(gt_class, seg_class))

def measured_distance(self):
    """
    This functions calculates the average symmetric distance and the
    hausdorff distance between a prediction and a reference image

    :return: hausdorff distance and average symmetric distance, hausdorff distance at perc
    and masd
    """
    if "hd_perc" in self.dict_args.keys():
        perc = self.dict_args["hd_perc"]
    else:
        perc = 95
    if np.sum(self.pred + self.ref) == 0:
        return 0, 0, 0, 0
    (
        ref_border_dist,
        pred_border_dist,
        ref_border,
        pred_border,
    ) = self.border_distance()
    print(ref_border_dist)
    average_distance = (np.sum(ref_border_dist) + np.sum(pred_border_dist)) / (
        np.sum(pred_border + ref_border)
    )
    masd = 0.5 * (
        np.sum(ref_border_dist) / np.sum(pred_border)
        + np.sum(pred_border_dist) / np.sum(ref_border)
    )

    hausdorff_distance = np.max([np.max(ref_border_dist), np.max(pred_border_dist)])
    
    hausdorff_distance_perc = np.max(
        [
            np.percentile(ref_border_dist[pred_border > 0], q=perc),
            np.percentile(pred_border_dist[ref_border > 0], q=perc),
        ]
    )


    return hausdorff_distance, average_distance, hausdorff_distance_perc, masd

def measured_hausdorff_distance(self):
    return self.measured_distance()[0]

def measured_average_distance(self):
    return self.measured_distance()[1]

def measured_masd(self):
    return self.measured_distance()[3]

def measured_hausdorff_distance_perc(self):
    return self.measured_distance()[2]

def conformity_coefficient(seg_class, gt_class):
    true_positives = tp(gt_class, seg_class)
    false_positives = fp(gt_class, seg_class)
    false_negatives = fn(gt_class, seg_class)
    
    return 1 - (false_positives + false_negatives)/true_positives
    
def sensitivity(seg_class, gt_class):
    n_pos_ref_val = n_pos_ref(gt_class)
    
    if n_pos_ref_val == 0:
        warnings.warn("Reference empty, sensitivity not defined")
        return np.nan
    true_positives = np.sum(np.logical_and(seg_class == 1, gt_class == 1))
    return true_positives / n_pos_ref_val

def specificity(gt_class, seg_class):
    n_neg_ref_val = n_neg_ref(gt_class)
    if n_neg_ref_val == 0:
        warnings.warn("Reference all positive, specificity not defined")
        return np.nan
    true_negatives = np.sum(np.logical_and(seg_class == 0, gt_class == 0))
    return true_negatives / n_neg_ref_val

def positive_predictive_values(seg_class, gt_class):
    if np.sum(seg_class) == 0:
        if np.sum(gt_class) == 0:
            warnings.warn("Reference and prediction are empty, PPV not defined")
            return np.nan
        else:
            warnings.warn("Prediction is empty, PPV not defined but set to 0")
            return 0
    return np.sum(tp_map(gt_class, seg_class)) / (np.sum(tp_map(gt_class, seg_class)) + np.sum(fp_map(gt_class, seg_class)))

def volume_ratio(seg_class, gt_class):
    return n_pos_pred(seg_class)/n_pos_ref(gt_class)


#%% Define final output function
def calculate_metrics_for_classes(seg_path, gt_path, img_path, num_classes=len(tissue_labels)):
    seg_img = nib.load(seg_path)
    gt_img = nib.load(gt_path)
    img = nib.load(img_path)
    
    seg_data = seg_img.get_fdata()
    gt_data = gt_img.get_fdata()
    
    metrics_dict = {
        'DSC': [],
        'NSD': [],
        'AVD': [],
        'IoU':[],
        'HD': [],
        'pHD': [],
        'AD': [],
        'MASD': [],
        'CC': [],
        'TPR': [],
        'TNR': [],
        'PPV': [],
        'VR': [],
        }
    
    for class_index in range (1, num_classes+1):
        seg_class = np.where(seg_data == class_index, 1, 0)
        gt_class = np.where(gt_data == class_index, 1, 0)
        
        DSC = dsc(seg_class, gt_class)
        NSD = normalised_surface_distance(seg_class, gt_class)
        AVD = absolute_volume_difference(seg_class, gt_class, img)
        IoU = intersection_over_union(seg_class, gt_class)
        CC = conformity_coefficient(seg_class, gt_class)
        TPR = sensitivity(seg_class, gt_class)
        TNR = specificity(gt_class, seg_class)
        PPV = positive_predictive_values(seg_class, gt_class)
        VR = volume_ratio(seg_class, gt_class)
        
        metrics_dict['DSC'].append(DSC)
        metrics_dict['NSD'].append(NSD)
        metrics_dict['AVD'].append(AVD)
        metrics_dict['IoU'].append(IoU)
        metrics_dict['HD'].append(HD)
        metrics_dict['pHD'].append(pHD)
        metrics_dict['AD'].append(AD)
        metrics_dict['MASD'].append(MASD)
        metrics_dict['CC'].append(CC)
        metrics_dict['TPR'].append(TPR)
        metrics_dict['TNR'].append(TNR)
        metrics_dict['PPV'].append(PPV)
        metrics_dict['VR'].append(VR)
        
    return metrics_dict        
    
#%% Run
def main():
    script_path = os.path.abspath(__file__)
    print(f"The absolute path of the script is: {script_path}")

    parser = get_parser()
    args = parser.parse_args()
    
    #%% Set directories
    input_dir = args.input_dir
    base_name = os.path.basename(args.input_dir)

    # Split the name from its extension
    name, ext = os.path.splitext(base_name)

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
    
    tau_value = 2 # imprecision value allowed for surface distance between boundaries

    image_paths = sorted(
        glob(os.path.join(input_dir, 'images', '*.nii.gz'), recursive=True))

    mm_seg_paths = sorted(
        glob(os.path.join(input_dir, 'mm', '*.nii.gz'), recursive=True))
    
    gt_seg_paths = sorted(
        glob(os.path.join(input_dir, 'gt', '*.nii.gz'), recursive=True))

    results_dict = {}

    for index, file in enumerate(image_paths):
        match = re.search('_(\d+).nii.gz', file)

        if match:
            subject_id = match.group(1)
            temporary_results = {'ID': subject_id}

            image_path = file

            mm_seg_matches = [mm for mm in mm_seg_paths if f'{subject_id}.nii.gz' in mm]
            if len(mm_seg_matches) == 0:
                raise FileNotFoundError(f"No mm segmentation file found for subject {subject_id}")
            elif len(mm_seg_matches) > 1:
                raise ValueError(f"More than one mm segmentation file found for subject {subject_id}")
            mm_seg_path = mm_seg_matches[0]

            gt_seg_matches = [gt for gt in gt_seg_paths if f'{subject_id}.nii.gz' in gt]
            if len(gt_seg_matches) == 0:
                raise FileNotFoundError(f"No gt segmentation file found for subject {subject_id}")
            elif len(gt_seg_matches) > 1:
                raise ValueError(f"More than one gt segmentation file found for subject {subject_id}")
            gt_seg_path = gt_seg_matches[0] 
        
            print(f"Subject ID: {subject_id}")

            metrics_dict = calculate_metrics_for_classes(mm_seg_path, gt_seg_path, image_path)
            
            # Need assistance assigning tissue_labels based on... region? The order may differ between gt files of our users segmentation files though
            for metric_name, values in metrics_dict.items():
                for tissue, labels in tissue_labels:
                    temporary_results[f'{metric_name}_{tissue}'] = values[labels] if labels < len(values) else None
                    
                    print(f"Tissue: {tissue}, Labels: {labels}, {metric_name}: {temporary_results[f'{metric_name}_{tissue}']}")
                    
            all_subject_data.append(temporary_results)
            
    results = pd.DataFrame(all_subject_data)

    #%% Save
    results.to_csv(os.path.join(output_dir, 'segment_performance_metrics.csv'), index=False)

if __name__== "__main__":
    main()