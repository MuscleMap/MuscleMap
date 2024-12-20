#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For usage, type: python mm_evaluate_segmentations.py 

Author: Brian Kim

"""

#%% Import packages
import argparse
import warnings
import numpy as np
import nibabel as nib
import os
import re
import pandas as pd
from scipy import ndimage
import sys
from tqdm import tqdm

print("Command line arguments received:", sys.argv)

#%% Set command line argument function
def get_parser():
    parser = argparse.ArgumentParser(
        description="Calculate image segmentation performance metrics.")
    
    # Required arguments
    required = parser.add_argument_group("Required")
    
    required.add_argument("-g", '--ground_truth', required=True, type=str,
                          help="Path to ground truth labelmap.")
    
    required.add_argument("-s", '--segmentation', required=True, type=str,
                          help="Path to segmentation labelmap.")
    
    # Optional arguments
    optional = parser.add_argument_group("Optional")
    required.add_argument("-o", '--output_dir', required=False, type=str,
                            help="Output directory to save the results, output file name suffix = csv. If left empty, saves to current working directory.")
    
    return parser

#%% Define basic overlap functions | gt = Ground Truth segmentation & seg = comparison segmentation
def compute_confusion_matrix(gt_class, seg_class):
    """
    Compute the confusion matrix for binary segmentation masks using element-wise operations.
    
    Parameters:
        gt_class (ndarray): Ground truth binary mask (0s and 1s).
        seg_class (ndarray): Segmentation binary mask (0s and 1s).
    
    Returns:
        dict: A dictionary containing the following confusion matrix values:
            - "tp": True positives
            - "fp": False positives
            - "tn": True negatives
            - "fn": False negatives
    """
    if gt_class.shape != seg_class.shape:
        raise ValueError("Ground truth and segmentation masks must have the same shape.")
    
    # Optimise memory by storing binary label values as integers
    gt_class = gt_class.astype(np.uint8)
    seg_class = seg_class.astype(np.uint8)

    tp = np.sum(gt_class * seg_class, dtype=np.float32)
    fp = np.sum((1 - gt_class) * seg_class, dtype=np.float32)
    tn = np.sum((1 - gt_class) * (1 - seg_class), dtype=np.float32)
    fn = np.sum(gt_class * (1 - seg_class), dtype=np.float32)
    
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

"""
The following functions adapted from MONAI/MetricsReloaded: [border_map, border_map2, foreground_component, 
list_foreground_component, border_distance, normalised_surface_distance, measured_distance].

Maier-Hein, L., Reinke, A., Godau, P., et al. (2024). Metrics reloaded: recommendations for image analysis validation.
Nature Methods, 21(3), 195–212. https://doi.org/10.1038/s41592-023-02151-z

Changes to original code:
- Converted MorphologyOps class to global function
- Amended function args to be binary labelmaps

Original Copyright:
Copyright (c) Carole Sudre

Licensed under the Apache License, Version 2.0 (the "License");
You may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""
def border_map(binary_map, connectivity):
    """
    Create the border map defined as the difference between the original image 
    and its eroded version.

    :param binary_map: Binary image map.
    :param connectivity: Connectivity value.
    :return: Border map.
    """
    eroded = ndimage.binary_erosion(binary_map, structure=np.ones((3, 3, 3)) if len(binary_map.shape) == 3 else None)
    border = binary_map - eroded
    return border

def border_map2(binary_map, connectivity):
    """
    Creates the border for a 3D image.

    :param binary_map: Binary image map.
    :param connectivity: Connectivity value.
    :return: Border map.
    """
    west = ndimage.shift(binary_map, [-1, 0, 0], order=0)
    east = ndimage.shift(binary_map, [1, 0, 0], order=0)
    north = ndimage.shift(binary_map, [0, 1, 0], order=0)
    south = ndimage.shift(binary_map, [0, -1, 0], order=0)
    top = ndimage.shift(binary_map, [0, 0, 1], order=0)
    bottom = ndimage.shift(binary_map, [0, 0, -1], order=0)
    cumulative = west + east + north + south + top + bottom
    border = ((cumulative < 6) * binary_map) == 1
    return border

def foreground_component(binary_map):
    """
    Returns the foreground component in a labeled format.

    :param binary_map: Binary image map.
    :return: Labeled foreground component.
    """
    return ndimage.label(binary_map)

def list_foreground_component(binary_map):
    """
    Lists the components, volumes, and center of mass of foreground components.

    :param binary_map: Binary image map.
    :return: List of labeled components, volumes, and center of mass.
    """
    labels, _ = foreground_component(binary_map)
    list_ind_lab = []
    list_volumes = []
    list_com = []
    list_values = np.unique(labels)
    for f in list_values:
        if f > 0:
            tmp_lab = np.where(
                labels == f, np.ones_like(labels), np.zeros_like(labels)
            )
            list_ind_lab.append(tmp_lab)
            list_volumes.append(np.sum(tmp_lab))
            list_com.append(ndimage.center_of_mass(tmp_lab))
    return list_ind_lab, list_volumes, list_com

def border_distance(seg_class, gt_class, connectivity_type=1, pixdim=None):
    """
    This function determines the map of distance from the borders of the prediction (s)
    and the reference (gt) and the border maps themselves.

    :param seg_class: Segmentation class.
    :param gt_class: Ground truth class.
    :param connectivity_type: Connectivity type.
    :param pixdim: Pixel dimensions for distance transform.
    :return: Distance from borders for gt and seg, and the borders themselves.
    """
    border_gt = border_map(gt_class, connectivity_type)
    border_seg = border_map(seg_class, connectivity_type)
    distance_gt = ndimage.distance_transform_edt(
        1 - border_gt, sampling=pixdim
    )
    distance_seg = ndimage.distance_transform_edt(
        1 - border_seg, sampling=pixdim
    )
    distance_border_seg = border_gt * distance_seg
    distance_border_gt = border_seg * distance_gt
    return distance_border_gt, distance_border_seg, border_gt, border_seg

def normalised_surface_distance(seg_class, gt_class, dict_args={}):
    """
    Calculates the normalised surface distance (NSD) between prediction (s) and reference (gt).

    :param seg_class: Segmentation class.
    :param gt_class: Ground truth class.
    :param dict_args: Dictionary of arguments, with the 'nsd' key defining the tau parameter.
    :return: Normalised surface distance.
    """
    tau = dict_args.get("nsd", 1)
    dist_gt, dist_seg, border_gt, border_seg = border_distance(seg_class, gt_class)
    reg_gt = np.where(
        dist_gt <= tau, np.ones_like(dist_gt), np.zeros_like(dist_gt)
    )
    reg_seg = np.where(
        dist_seg <= tau, np.ones_like(dist_seg), np.zeros_like(dist_seg)
    )
    numerator = np.sum(border_seg * reg_gt) + np.sum(border_gt * reg_seg)
    denominator = np.sum(border_gt) + np.sum(border_seg)
    return numerator / denominator

def measured_distance(seg_class, gt_class, dict_args={}):
    """
    Calculates the average symmetric distance and the hausdorff distance between the gt and segmentation.

    :param seg_class: Segmentation class.
    :param gt_class: Ground truth class.
    :param dict_args: Dictionary of arguments, with the 'hd_perc' key for the Hausdorff distance percentile.
    :return: Hausdorff distance, average symmetric distance, and the Hausdorff distance at percentile.
    """
    perc = dict_args.get("hd_perc", 95)
    if np.sum(seg_class + gt_class) == 0:
        return 0, 0, 0, 0
    (
        gt_border_dist,
        seg_border_dist,
        gt_border,
        seg_border,
    ) = border_distance(seg_class, gt_class)

    average_distance = (np.sum(gt_border_dist) + np.sum(seg_border_dist)) / (
        np.sum(seg_border + gt_border)
    )
    masd = 0.5 * (
        np.sum(gt_border_dist) / np.sum(seg_border)
        + np.sum(seg_border_dist) / np.sum(gt_border)
    )

    hausdorff_distance = np.max([np.max(gt_border_dist), np.max(seg_border_dist)])
    
    hausdorff_distance_perc = np.max(
        [
            np.percentile(gt_border_dist[seg_border > 0], q=perc),
            np.percentile(seg_border_dist[gt_border > 0], q=perc),
        ]
    )

    return hausdorff_distance, average_distance, hausdorff_distance_perc, masd

def absolute_volume_difference(seg_class, gt_class, seg_path): 
    # Load the NIfTI images
    seg_img = nib.load(seg_path) # have to load separately to use pixdims for volume
    
    # Access the headers for pixel dimensions
    seg_header = seg_img.header
    
    # Extract pixel dimensions (voxel size)
    pixdim_x, pixdim_y, pixdim_z = seg_header['pixdim'][1:4]
    
    dim_x, dim_y, dim_z = seg_img.header['dim'][1:4]
    pixdim_x, pixdim_y, pixdim_z = seg_img.header['pixdim'][1:4]
    
    seg_volume = np.around(seg_class.sum() * pixdim_x * pixdim_y * pixdim_z / 1000, decimals=2)
    gt_volume = np.around(gt_class.sum() * pixdim_x * pixdim_y * pixdim_z / 1000, decimals=2)
    
    avd = abs(seg_volume - gt_volume)
  
    return avd

def percentage_volume_difference(seg_class, gt_class, seg_path): 
    # Load the NIfTI images
    seg_img = nib.load(seg_path) # have to load separately to use pixdims for volume
    
    # Access the headers for pixel dimensions
    seg_header = seg_img.header
    
    # Extract pixel dimensions (voxel size)
    pixdim_x, pixdim_y, pixdim_z = seg_header['pixdim'][1:4]
    
    dim_x, dim_y, dim_z = seg_img.header['dim'][1:4]
    pixdim_x, pixdim_y, pixdim_z = seg_img.header['pixdim'][1:4]
    
    seg_volume = np.around(seg_class.sum() * pixdim_x * pixdim_y * pixdim_z / 1000, decimals=2)
    gt_volume = np.around(gt_class.sum() * pixdim_x * pixdim_y * pixdim_z / 1000, decimals=2)
    
    pvd = 100*abs(seg_volume - gt_volume)/(gt_volume)
  
    return pvd

def dsc(conf_matrix):
    numerator = 2 * conf_matrix["tp"]
    denominator = 2 * conf_matrix["tp"] + conf_matrix["fp"] + conf_matrix["fn"]
    if denominator == 0:
        warnings.warn("Both gt and s are empty - set to 1 as correct solution even if not defined")
        return 1
    return numerator / denominator

def intersection_over_union(conf_matrix):
    if conf_matrix["tp"] + conf_matrix["fp"] + conf_matrix["fn"] == 0:
        warnings.warn("Both gt and s are empty")
        return np.nan
    return conf_matrix["tp"] / (conf_matrix["tp"] + conf_matrix["fp"] + conf_matrix["fn"])

def conformity_coefficient(conf_matrix):
    if conf_matrix["tp"] == 0:
        warnings.warn("True positives are zero, conformity coefficient not defined")
        return np.nan
    return 1 - (conf_matrix["fp"] + conf_matrix["fn"]) / conf_matrix["tp"]

def sensitivity(conf_matrix, n_pos_gt_val):
    if n_pos_gt_val == 0:
        warnings.warn("Reference (gt) empty, sensitivity not defined")
        return np.nan
    return conf_matrix["tp"] / n_pos_gt_val

def specificity(conf_matrix, n_neg_gt_val):
    if n_neg_gt_val == 0:
        warnings.warn("Reference (gt) all positive, specificity not defined")
        return np.nan
    return conf_matrix["tn"] / n_neg_gt_val

def positive_predictive_values(conf_matrix):
    if conf_matrix["tp"] + conf_matrix["fp"] == 0:
        warnings.warn("Prediction (s) is empty, PPV not defined")
        return np.nan
    return conf_matrix["tp"] / (conf_matrix["tp"] + conf_matrix["fp"])

def volume_ratio(seg_class, gt_class):
    n_pos_seg = np.sum(seg_class)
    n_pos_gt = np.sum(gt_class)
    if n_pos_gt == 0:
        warnings.warn("Reference (gt) empty, volume ratio not defined")
        return np.nan
    return n_pos_seg / n_pos_gt

#%% Main Metrics Calculation
def calculate_metrics_for_classes(seg_path, gt_path, num_classes):
    seg_img = nib.load(seg_path)
    gt_img = nib.load(gt_path)
    
    seg_data = seg_img.get_fdata()
    gt_data = gt_img.get_fdata()
    
    metrics_dict = {label: {} for label in range(1, num_classes + 1)}
    
    for label in tqdm(range(1, num_classes + 1), desc="Calculating Metrics", unit="class"):
        seg_class = (seg_data == label).astype(np.uint8)
        gt_class = (gt_data == label).astype(np.uint8)

        # Compute confusion matrix
        conf_matrix = compute_confusion_matrix(gt_class, seg_class)

        # Derived values
        n_pos_gt = np.sum(gt_class)
        n_neg_gt = np.prod(gt_class.shape) - n_pos_gt

        # Calculate metrics
        metrics_dict[label]['DSC'] = dsc(conf_matrix)
        metrics_dict[label]['NSD'] = normalised_surface_distance(seg_class, gt_class)
        metrics_dict[label]['AVD'] = absolute_volume_difference(seg_class, gt_class, seg_path)
        metrics_dict[label]['PVD'] = percentage_volume_difference(seg_class, gt_class, seg_path)
        metrics_dict[label]['IoU'] = intersection_over_union(conf_matrix)
        metrics_dict[label]['HD'] = measured_distance(seg_class, gt_class)[0]
        metrics_dict[label]['pHD'] = measured_distance(seg_class, gt_class)[2]
        metrics_dict[label]['AD'] = measured_distance(seg_class, gt_class)[1]
        metrics_dict[label]['MASD'] = measured_distance(seg_class, gt_class)[3]
        metrics_dict[label]['CC'] = conformity_coefficient(conf_matrix)
        metrics_dict[label]['TPR'] = sensitivity(conf_matrix, n_pos_gt)
        metrics_dict[label]['TNR'] = specificity(conf_matrix, n_neg_gt)
        metrics_dict[label]['PPV'] = positive_predictive_values(conf_matrix)
        metrics_dict[label]['VR'] = volume_ratio(seg_class, gt_class)

    return metrics_dict

#%% Initialize results table
results = pd.DataFrame()

#%% Main function
def main():
    script_path = os.path.abspath(__file__)
    print(f"The absolute path of the script is: {script_path}")

    parser = get_parser()
    args = parser.parse_args()
    
    seg_path = args.segmentation
    gt_path = args.ground_truth
    
    if args.output_dir is None:
        output_dir = os.getcwd()
    elif not os.path.exists(args.output_dir):
        output_dir = os.path.abspath(args.output_dir)
        os.makedirs(output_dir)
    else:
        output_dir = args.output_dir
    
    match = re.search(r'_(\d+)(?=\.[^.]+$)', seg_path)
    if match:
        ID = match.group(1)
        output_filename = f'{ID}_evaluate_segmentations_metrics.csv'
    else:
        output_filename = 'evaluate_segmentations_metrics.csv'

    # Determine number of classes dynamically
    seg_img = nib.load(seg_path)
    num_classes = int(np.max(seg_img.get_fdata()))

    # Calculate metrics
    results_dict = calculate_metrics_for_classes(seg_path, gt_path, num_classes)

    # Convert the metrics dictionary to a DataFrame
    results_df = pd.DataFrame.from_dict(results_dict, orient='index')

    # Save the DataFrame to a CSV file
    results_df.to_csv(os.path.join(output_dir, output_filename), index_label="Label")

#%% Run
if __name__== "__main__":
    main()