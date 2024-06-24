#!/usr/bin/env python
# -*- coding: utf-8

# For usage, type: python mm_segment.py -h

# Authors: Richard Yin, Eddo Wesselink, and Kenneth Weber

#IMPORTS: necessary libraries, modules, including MONAI for image processing, argparse, and torch for Deep Learning
import argparse
import logging
import os
import sys
import glob
from monai.inferers import SliceInferer
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscreted,
    Compose,
    Invertd,
    CropForegroundd, 
    LoadImaged,
    Orientationd,
    Spacingd,
    EnsureTyped,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    FillHolesd,
    SaveImaged,
    KeepLargestConnectedComponentd,
)
from monai.networks.layers import Norm
from monai.utils import set_determinism
from monai.data import (
    Dataset,
    DataLoader,
    decollate_batch,
)
from mm_segmentUtil import check_image_exists, get_model_and_config_paths, load_model_config, validate_arguments
import torch

#get_parser: parses command line a0rguemnts, sets up a) required (image, body region), and b) optional arguments (model, output file name)
def get_parser():
    parser = argparse.ArgumentParser(
        description="Segment an input image according to the specified deep learning model.")
    
    #Required arguments
    required = parser.add_argument_group("Required")
    required.add_argument("-i", '--image', required=True, type=str,
                        help="Image to segment. Can be multiple images separated with spaces.")
    required.add_argument("-r", '--region', required=True, type=str,
                        help="Body region of input image.")
    
    
    #Optional arguments
    optional = parser.add_argument_group("Optional")
    optional.add_argument("-m", '--model', default=None, required=False, type=str,
                        help="Option to specifiy another model.")
    optional.add_argument("-o", '--output_file_name', default='image_dseg.nii.gz', required=False, type=str,
                        help="Output file name. By default, dseg suffix will be added, and the output extension will be .nii.gz.")
    return parser





#main: parses command-line arguments using parser, runs model, inference, post-processing #Richard Add onto this description


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = get_parser()
    args = parser.parse_args()

    # Validate Arguments
    validate_arguments(args)

    # Check that the image exists and is readable
    logging.info(f"Checking if image '{args.i}' exists and is readable...")
    check_image_exists(args.i)

    # Load model configuration
    logging.info("Loading configuration file...")

    # Get model and config paths
    model_path, model_config_path = get_model_and_config_paths(args.region, args.model)

    # Load model configuration
    model_config = load_model_config(model_config_path)

    try:
        roi_size = tuple(model_config['parameters']['roi_size'])
        spatial_window_batch_size = model_config['parameters']['spatial_window_batch_size']
        amount_of_labels = model_config['parameters']['amount_of_labels']
        pix_dim = tuple(model_config['parameters']['pix_dim'])
        model_continue_training = model_config['parameters']['model_continue_training']
    except KeyError as e:
        logging.error(f"Missing key in model configuration file: {e}")
        sys.exit(1)    

    #directory setup
    data_dir = args.image
    images = sorted(glob.glob(os.path.join(data_dir, "testing", "*img.nii.gz"))) #KEN Start with just one image but later we can allow them to specificy multiple images
    test_files = [{"image": img} for img in zip(images)] #KEN Remove test and validation it's now an input and output





    #use os.path.join for all path constructions to ensure cross platofrm compatability

    # set some important parameters comparable to training #KEN see if these can go in the parameters file
    #Richard: MOVE hardcoded parameters to a config file
    #confirm whether setting the seed for reproducibility is necessary for inference
    #set seed for reproducibility (identical to training part)
    #set_determinism(seed=0) #Ken Don't belive this is necessary for inference



    #create transforms identical to training part, but here we don't specifiy the label key 
    #Ensure all necessary transformations are included based on the model requirements.

    inference_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(
            keys=["image"],
            pixdim=pix_dim,
            mode=("bilinear")),
        Orientationd(keys=["image"], axcodes="RAS"),
        CropForegroundd(keys=["image"], source_key="image"),
        NormalizeIntensityd(keys=["image"], nonzero = True),
        EnsureTyped(keys=["image"])
    ])

    #create iterable dataset and dataloader, identical to training part
    inference_transforms_dataset = Dataset(
    data=test_files, transform=inference_transforms,
    )

    inference_transforms_loader = DataLoader(
    inference_transforms_dataset , batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    #device config #KEN probably want to keep
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    #Richard: Move labels to config file
    #Ensure flexibility to handle models with varying label counts
    #Validate post processing steps based on specific requiremetns of different models

    #set post transforms #KEN can probably go into parameters file
    post_transforms = Compose([
    Invertd(
        keys="pred",
        transform=inference_transforms,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=False,
        to_tensor=True,
        device = device
    ),
    AsDiscreted(keys="pred", argmax=True),
    FillHolesd(keys="pred", applied_labels=[1,2,3,4,5,6,7,8,9]), #Ken Not all Models will have 1-9 labels
    KeepLargestConnectedComponentd(keys="pred", applied_labels=[1,2,3,4,5,6,7,8,9]), #Ken Not all Models will have 1-9 labels
    SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=save_dir, output_dtype=('int16'), separate_folder = False, resample=False)
    ])



    #Richard: Move model params to config file
    #error handling for model
    #Create model and set parameters. #Ken Goes in the Paramters Folder
    model= UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=amount_of_labels,
    channels=(64, 128, 256, 512, 1024),
    act= 'LeakyRelu',
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.INSTANCE,
    ).to(device)

    #load pre-excisting model if we want to continue training
    model.load_state_dict(torch.load(model_path,
    model.eval()





    #Run inference on the image and images using model, post-processes the predictions
    #Richard: add logging for inference steps
    #Exception handling for inference

    #Inference part
    with torch.no_grad():
        for i, input_data in enumerate(inference_transforms_loader):
            val_inputs = input_data["image"].to(device)
            axial_inferer = SliceInferer(roi_size=roi_size, sw_batch_size=spatial_window_batch_size, spatial_dim=2)
            input_data["pred"] = axial_inferer(val_inputs, model)
            val_data = [post_transforms(i) for i in decollate_batch(input_data)]




    """Optional"""
    #Allow list of filenames separated by spaces
    #Make sure relative and absolute paths work
    

if __name__ == "__main__":
    main()
    