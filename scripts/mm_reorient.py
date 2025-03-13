import os
import re
import json
import logging
import numpy as np
import nibabel as nib
import argparse
from tqdm import tqdm
from glob import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mri_reorient.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_subject_modality(filename):
    match = re.match(r"(sub-\d+)_ses-shoulder_([^_.]+)", filename)
    return match.groups() if match else (None, None)

def get_reorientation_order(pixel_dims):
    """ Ensures that the largest pixel dimension is the 3rd axis. """
    sorted_indices = np.argsort(pixel_dims)  # Sort indices by pixel dimension
    largest_dim = sorted_indices[-1]  # Index of the largest dimension

    if largest_dim == 2:
        return (0, 1, 2)
    elif largest_dim == 0:
        return (1, 2, 0)
    else:
        return (0, 2, 1)

def reorient_to_in_plane_high_res(image_path, output_path, is_label=False, ref_affine=None, ref_zooms=None):
    img = nib.load(image_path)
    affine = img.affine
    header = img.header
    pixel_dimensions = header.get_zooms()
    
    data = np.asarray(img.dataobj).astype(np.int32) if is_label else img.get_fdata()
    if len(pixel_dimensions) != 3:
        raise ValueError(f"Expected 3D pixel dimensions, got {pixel_dimensions}")
    
    target_order = get_reorientation_order(pixel_dimensions)
    
    if target_order == (0, 1, 2):  # No reorientation needed
        return None
    
    logger.info(f"Reorienting {image_path}")
    logger.info(f"Original shape: {data.shape}, Pixel dims: {pixel_dimensions}")
    logger.info(f"Original affine:\n{affine}")

    # Apply reorientation
    reoriented_data = np.transpose(data, axes=target_order)
    new_spacing = tuple(pixel_dimensions[i] for i in target_order)

    # Construct the reorientation matrix
    P = np.eye(4)
    for i, idx in enumerate(target_order):
        P[i, idx] = 1

    new_affine = affine @ P.T  # Matrix multiplication to get new affine

    # Ensure label matches image affine
    if is_label and ref_affine is not None and ref_zooms is not None:
        if not np.allclose(new_affine, ref_affine, atol=1e-4):
            logger.warning(f"Affine mismatch for label: {image_path}, using reference affine")
        new_affine = ref_affine
        new_spacing = ref_zooms

    logger.info(f"New shape: {reoriented_data.shape}, New pixel dims: {new_spacing}")
    logger.info(f"New affine:\n{new_affine}")

    # Save reoriented image
    new_img = nib.Nifti1Image(np.round(reoriented_data).astype(np.int32) if is_label else reoriented_data, new_affine)
    new_img.header.set_sform(new_affine, code=4)
    new_img.header.set_qform(new_affine, code=4)
    new_img.header.set_zooms(new_spacing)

    nib.save(new_img, output_path)
    logger.info(f"Saved reoriented image to {output_path}")

    return reoriented_data.shape, new_affine, new_spacing

def main(input_dir, output_dir, json_file=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir), 'data_reoriented')
    os.makedirs(output_dir, exist_ok=True)
    
    ids = json.load(open(json_file))['ids'] if json_file else None
    
    image_paths = glob(os.path.join(input_dir, 'sourcedata', '*', 'ses-shoulder', 'anat', '*.nii.gz'))
    label_paths = glob(os.path.join(input_dir, 'derivatives', 'labels', '*', 'ses-shoulder', 'anat', '*dseg.nii.gz'))
    
    image_dict = {extract_subject_modality(os.path.basename(img)): img for img in image_paths if extract_subject_modality(os.path.basename(img))[0] and (ids is None or extract_subject_modality(os.path.basename(img))[0] in ids)}
    label_dict = {extract_subject_modality(os.path.basename(lbl)): lbl for lbl in label_paths if extract_subject_modality(os.path.basename(lbl))[0] and (ids is None or extract_subject_modality(os.path.basename(lbl))[0] in ids)}
    
    train_files = [{"image": image_dict[k], "label": label_dict[k]} for k in image_dict if k in label_dict]
    if not train_files:
        raise ValueError("No matching image-label pairs found")
    
    for pair in tqdm(train_files, desc='Reorienting MRI to in-plane high-resolution axes'):
        image_output_path = os.path.join(output_dir, os.path.relpath(pair["image"], input_dir))
        label_output_path = os.path.join(output_dir, os.path.relpath(pair["label"], input_dir))
        os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(label_output_path), exist_ok=True)
        
        img_result = reorient_to_in_plane_high_res(pair["image"], image_output_path)
        if img_result:  # Process label only if reorientation occurred
            img_shape, img_affine, img_zooms = img_result
            reorient_to_in_plane_high_res(pair["label"], label_output_path, is_label=True, ref_affine=img_affine, ref_zooms=img_zooms)
