#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For usage, type: python mm_segment_performance_metrics.py 

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
                          help="Path to ground truth segmentation labelmap.")
    
    required.add_argument("-m", '--muscle_map', required=True, type=str,
                          help="Path to MuscleMap segmentation labelmap.")
    
    # Optional arguments
    optional = parser.add_argument_group("Optional")
    required.add_argument("-o", '--output_dir', required=False, type=str,
                            help="Output directory to save the results, output file name suffix = csv. If left empty, saves to current working directory.")
    
    return parser

#%% Define basic overlap functions | gt = Ground Truth segmentation & mm = MuscleMap segmentation
def fp_map(gt, mm):
    gt_float = np.asarray(gt, dtype=np.float32)
    mm_float = np.asarray(mm, dtype=np.float32)
    return np.asarray((mm_float - gt_float) > 0.0, dtype=np.float32)

def fn_map(gt, mm):
    gt_float = np.asarray(gt, dtype=np.float32)
    mm_float = np.asarray(mm, dtype=np.float32)
    return np.asarray((gt_float - mm_float) > 0.0, dtype=np.float32)

def tp_map(gt, mm):
    gt_float = np.asarray(gt, dtype=np.float32)
    mm_float = np.asarray(mm, dtype=np.float32)
    return np.asarray((gt_float + mm_float) > 1.0, dtype=np.float32)

def tn_map(gt, mm):
    gt_float = np.asarray(gt, dtype=np.float32)
    mm_float = np.asarray(mm, dtype=np.float32)
    return np.asarray((gt_float + mm_float) < 0.5, dtype=np.float32)

def union_map(gt, mm):
    return np.asarray((gt + mm) > 0.5, dtype=np.float32)

def intersection_map(gt, mm):
    return np.multiply(gt, mm)

def n_pos_gt(gt):
    return np.sum(gt)

def n_neg_gt(gt):
    return np.sum(1 - gt)

def n_pos_mm(mm):
    """
    Returns the number of positive elements in the mmiction
    """
    return np.sum(mm)

def n_neg_mm(mm):
    """
    Returns the number of negative elements in the mmiction
    """
    return np.sum(1 - mm)

def fp(gt, mm):
    """
    Calculates the number of FP as sum of elements in FP_map
    """
    return np.sum(fp_map(gt, mm))

def fn(gt, mm):
    """
    Calculates the number of FN as sum of elements of FN_map
    """
    return np.sum(fn_map(gt, mm))

def tp(gt, mm):
    """
    Returns the number of true positive (TP) elements
    """
    return np.sum(tp_map(gt, mm))

def tn(gt, mm):
    """
    Returns the number of True Negative (TN) elements
    """
    return np.sum(tn_map(gt, mm))

def n_intersection(gt, mm):
    """
    Returns the number of elements in the intersection of gterence and mmiction (=TP)
    """
    return np.sum(intersection_map(gt, mm))

def n_union(gt, mm):
    """
    Returns the number of elements in the union of gterence and mmiction

    .. math::

        U = {\vert} mm {\vert} + {\vert} gt {\vert} - TP

    """
    return np.sum(union_map(gt, mm))

def border_distance(mm_class, gt_class, connectivity_type=1, pixdim=None):
    """
    This function determines the map of distance from the borders of the
    mmiction and the gterence and the border maps themselves

    :return: distance_border_gt, distance_border_mm, border_gt, border_mm
    """
    border_gt = MorphologyOps(gt_class, connectivity_type).border_map()
    border_mm = MorphologyOps(mm_class, connectivity_type).border_map()
    distance_gt = ndimage.distance_transform_edt(
        1 - border_gt, sampling=pixdim
    )
    distance_mm = ndimage.distance_transform_edt(
        1 - border_mm, sampling=pixdim
    )
    distance_border_mm = border_gt * distance_mm
    distance_border_gt = border_mm * distance_gt
    return distance_border_gt, distance_border_mm, border_gt, border_mm

#%% Define performance metric functions
def dsc(mm_class, gt_class):

    numerator = 2 * np.sum(tp_map(gt_class, mm_class))
    denominator = np.sum(n_pos_mm(mm_class)) + np.sum(n_pos_gt(gt_class))
    if denominator == 0:
        warnings.warn("Both mmiction and gterence are empty - set to 1 as correct solution even if not defined")
        return 1
    else:
        return numerator / denominator

class MorphologyOps(object): # Can we move this to utils?
    """
    Class that performs the morphological operations needed to get notably
    connected component. To be used in the evaluation
    """

    def __init__(self, binary_img, connectivity):
        self.binary_map = np.asarray(binary_img, dtype=np.int8)
        self.connectivity = connectivity

    def border_map(self):
        """
        Create the border map defined as the difference between the original image 
        and its eroded version

        :return: border
        """
        eroded = ndimage.binary_erosion(self.binary_map)
        border = self.binary_map - eroded
        return border

    def border_map2(self):
        """
        Creates the border for a 3D image
        :return:
        """
        west = ndimage.shift(self.binary_map, [-1, 0, 0], order=0)
        east = ndimage.shift(self.binary_map, [1, 0, 0], order=0)
        north = ndimage.shift(self.binary_map, [0, 1, 0], order=0)
        south = ndimage.shift(self.binary_map, [0, -1, 0], order=0)
        top = ndimage.shift(self.binary_map, [0, 0, 1], order=0)
        bottom = ndimage.shift(self.binary_map, [0, 0, -1], order=0)
        cumulative = west + east + north + south + top + bottom
        border = ((cumulative < 6) * self.binary_map) == 1
        return border

    def foreground_component(self):
        return ndimage.label(self.binary_map)

    def list_foreground_component(self):
        labels, _ = self.foreground_component()
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
        
def normalised_surface_distance(mm_class, gt_class, dict_args={}):
    """
    Adapted from MONAI/MetricsReloaded.
    
    Calculates the normalised surface distance (NSD) between mmiction and gterence
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
        tau = 1
    dist_gt, dist_mm, border_gt, border_mm = border_distance(mm_class, gt_class)
    reg_gt = np.where(
        dist_gt <= tau, np.ones_like(dist_gt), np.zeros_like(dist_gt)
    )
    reg_mm = np.where(
        dist_mm <= tau, np.ones_like(dist_mm), np.zeros_like(dist_mm)
    )
    numerator = np.sum(border_mm * reg_gt) + np.sum(border_gt * reg_mm)
    denominator = np.sum(border_gt) + np.sum(border_mm)
    return numerator / denominator

def absolute_volume_difference(mm_class, gt_class, mm_path): 
    # Load the NIfTI images
    mm_img = nib.load(mm_path) # have to load separately to use pixdims for volume
    
    # Access the headers for pixel dimensions
    mm_header = mm_img.header
    
    # Extract pixel dimensions (voxel size)
    pixdim_x, pixdim_y, pixdim_z = mm_header['pixdim'][1:4]
    
    dim_x, dim_y, dim_z = mm_img.header['dim'][1:4]
    pixdim_x, pixdim_y, pixdim_z = mm_img.header['pixdim'][1:4]
    
    mm_volume = np.around(mm_class.sum() * pixdim_x * pixdim_y * pixdim_z / 1000, decimals=2)
    gt_volume = np.around(gt_class.sum() * pixdim_x * pixdim_y * pixdim_z / 1000, decimals=2)
    
    avd = abs(mm_volume - gt_volume)
  
    return avd

def percentage_volume_difference(mm_class, gt_class, mm_path): 
    # Load the NIfTI images
    mm_img = nib.load(mm_path) # have to load separately to use pixdims for volume
    
    # Access the headers for pixel dimensions
    mm_header = mm_img.header
    
    # Extract pixel dimensions (voxel size)
    pixdim_x, pixdim_y, pixdim_z = mm_header['pixdim'][1:4]
    
    dim_x, dim_y, dim_z = mm_img.header['dim'][1:4]
    pixdim_x, pixdim_y, pixdim_z = mm_img.header['pixdim'][1:4]
    
    mm_volume = np.around(mm_class.sum() * pixdim_x * pixdim_y * pixdim_z / 1000, decimals=2)
    gt_volume = np.around(gt_class.sum() * pixdim_x * pixdim_y * pixdim_z / 1000, decimals=2)
    
    pvd = 100*abs(mm_volume - gt_volume)/(gt_volume)
  
    return pvd

def intersection_over_union(mm_class, gt_class):
    if np.sum(mm_class) == 0 and np.sum(gt_class) == 0:
        warnings.warn("Both gt and mm are empty")
        return np.nan
    return np.sum(intersection_map(gt_class, mm_class)) / np.sum(union_map(gt_class, mm_class))

def measured_distance(mm_class, gt_class, dict_args={}):
    """
    This functions calculates the average symmetric distance and the
    hausdorff distance between the gt and mm segmentation

    :return: hausdorff distance and average symmetric distance, hausdorff distance at perc
    and masd
    """
    if "hd_perc" in dict_args.keys():
        perc = dict_args["hd_perc"]
    else:
        perc = 95
    if np.sum(mm_class + gt_class) == 0:
        return 0, 0, 0, 0
    (
        gt_border_dist,
        mm_border_dist,
        gt_border,
        mm_border,
    ) = border_distance(mm_class, gt_class)

    average_distance = (np.sum(gt_border_dist) + np.sum(mm_border_dist)) / (
        np.sum(mm_border + gt_border)
    )
    masd = 0.5 * (
        np.sum(gt_border_dist) / np.sum(mm_border)
        + np.sum(mm_border_dist) / np.sum(gt_border)
    )

    hausdorff_distance = np.max([np.max(gt_border_dist), np.max(mm_border_dist)])
    
    hausdorff_distance_perc = np.max(
        [
            np.percentile(gt_border_dist[mm_border > 0], q=perc),
            np.percentile(mm_border_dist[gt_border > 0], q=perc),
        ]
    )


    return hausdorff_distance, average_distance, hausdorff_distance_perc, masd

def conformity_coefficient(mm_class, gt_class):
    true_positives = tp(gt_class, mm_class)
    false_positives = fp(gt_class, mm_class)
    false_negatives = fn(gt_class, mm_class)
    
    return 1 - (false_positives + false_negatives)/true_positives
    
def sensitivity(mm_class, gt_class):
    n_pos_gt_val = n_pos_gt(gt_class)
    
    if n_pos_gt_val == 0:
        warnings.warn("gterence empty, sensitivity not defined")
        return np.nan
    true_positives = np.sum(np.logical_and(mm_class == 1, gt_class == 1))
    return true_positives / n_pos_gt_val

def specificity(mm_class, gt_class):
    n_neg_gt_val = n_neg_gt(gt_class)
    if n_neg_gt_val == 0:
        warnings.warn("gterence all positive, specificity not defined")
        return np.nan
    true_negatives = np.sum(np.logical_and(mm_class == 0, gt_class == 0))
    return true_negatives / n_neg_gt_val

def positive_predictive_values(mm_class, gt_class):
    if np.sum(mm_class) == 0:
        if np.sum(gt_class) == 0:
            warnings.warn("gterence and mmiction are empty, PPV not defined")
            return np.nan
        else:
            warnings.warn("mmiction is empty, PPV not defined but set to 0")
            return 0
    return np.sum(tp_map(gt_class, mm_class)) / (np.sum(tp_map(gt_class, mm_class)) + np.sum(fp_map(gt_class, mm_class)))

def volume_ratio(mm_class, gt_class):
    return n_pos_mm(mm_class)/n_pos_gt(gt_class)

#%% Define final output function
def calculate_metrics_for_classes(mm_path, gt_path, num_classes):
    mm_img = nib.load(mm_path)
    gt_img = nib.load(gt_path)
    
    mm_data = mm_img.get_fdata()
    gt_data = gt_img.get_fdata()
    
    # Dictionary where the key is the class index and the value is another dictionary of metrics
    metrics_dict = {class_index: {} for class_index in range(1, num_classes + 1)}
    
    for class_index in tqdm(range(1, num_classes + 1), desc="Calculating Metrics", unit="class"):
        mm_class = np.where(mm_data == class_index, 1, 0)
        gt_class = np.where(gt_data == class_index, 1, 0)
        
        # Calculate metrics
        metrics_dict[class_index]['DSC'] = dsc(mm_class, gt_class)
        metrics_dict[class_index]['NSD'] = normalised_surface_distance(mm_class, gt_class)
        metrics_dict[class_index]['AVD'] = absolute_volume_difference(mm_class, gt_class, mm_path)
        metrics_dict[class_index]['PVD'] = percentage_volume_difference(mm_class, gt_class, mm_path)
        metrics_dict[class_index]['IoU'] = intersection_over_union(mm_class, gt_class)
        metrics_dict[class_index]['HD'] = measured_distance(mm_class, gt_class)[0]
        metrics_dict[class_index]['pHD'] = measured_distance(mm_class, gt_class)[2]
        metrics_dict[class_index]['AD'] = measured_distance(mm_class, gt_class)[1]
        metrics_dict[class_index]['MASD'] = measured_distance(mm_class, gt_class)[3]
        metrics_dict[class_index]['CC'] = conformity_coefficient(mm_class, gt_class)
        metrics_dict[class_index]['TPR'] = sensitivity(mm_class, gt_class)
        metrics_dict[class_index]['TNR'] = specificity(mm_class, gt_class)
        metrics_dict[class_index]['PPV'] = positive_predictive_values(mm_class, gt_class)
        metrics_dict[class_index]['VR'] = volume_ratio(mm_class, gt_class)

    return metrics_dict


#%% Initialize results table
results = pd.DataFrame()

#%% Main function
def main():
    script_path = os.path.abspath(__file__)
    print(f"The absolute path of the script is: {script_path}")

    parser = get_parser()
    args = parser.parse_args()
    
    mm_path = args.muscle_map
    gt_path = args.ground_truth
    
    if args.output_dir is None:
        output_dir = os.getcwd()
    elif not os.path.exists(args.output_dir):
        output_dir = os.path.abspath(args.output_dir)
        os.makedirs(output_dir)
    else:
        output_dir = args.output_dir
    
    match = re.search(r'_(\d+)(?=\.[^.]+$)', mm_path)
    if match:
        ID = match.group(1)
        output_filename = f'{ID}_segment_performance_metrics.csv'
    else:
        output_filename = 'segment_performance_metrics.csv'

    # Determine number of classes dynamically
    mm_img = nib.load(mm_path)
    num_classes = int(np.max(mm_img.get_fdata()))

    # Calculate metrics
    results_dict = calculate_metrics_for_classes(mm_path, gt_path, num_classes)

    # Convert the metrics dictionary to a DataFrame
    results_df = pd.DataFrame.from_dict(results_dict, orient='index')

    # Save the DataFrame to a CSV file
    results_df.to_csv(os.path.join(output_dir, output_filename), index_label="Class_Index")

#%% Run
if __name__== "__main__":
    main()