#!/usr/bin/env python
# -*- coding: utf-8

# For usage, type: python mm_segment.py -h

# Authors: Richard Yin, Eddo Wesselink, and Kenneth Weber
import argparse
import os
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
    SaveImaged,
    FillHolesd,
    KeepLargestConnectedComponentd,
)
from monai.networks.layers import Norm
from monai.utils import set_determinism
from monai.data import (
    Dataset,
    DataLoader,
    decollate_batch,
)
import torch

def get_parser():
    parser = argparse.ArgumentParser(
        description="Segment an input image according to the specified deep learning model.")
    
    #Required arguments
    required = parser.add_argument_group("Required")
    required.add_argument("-i", required=True, type=str,
                        help="Image to segment. Can be multiple images separated with spaces.")
    required.add_argument("-r", required=True, type=str,
                        help="Body region of input image.")
    
    #Optional arguments
    optional = parser.add_argument_group("Optional")
    optional.add_argument("-m", default=None, required=False, type=str,
                        help="Option to specifiy another model.")
    optional.add_argument("-o", default=None, required=False, type=str,
                        help="Output file name. By default, dseg suffix will be added, and the output extension will be .nii.gz.")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    #Check that image exists and is readable

    #Check that body region and corresponding model are available
  

    #Load model and model parameters

    #setup data directory

    directory = os.environ.get("Monai_Directory") #Ken Assume OUTPUT is PWD os.get_pwd unles they specifcy /home/kenweber/input.nii.gz
    root_dir = tempfile.mkdtemp() if directory is None else directory
    save_dir = 'D:\\PS_Muscle_Segmentation\\Monai\\Current_projects\\QL\\CNN' #Assume is PWD os.get_pwd unles they specifcy /home/kenweber/input.nii.gz

    data_dir = 'D:\\PS_Muscle_Segmentation\\Monai\\Lx'
    images = sorted(glob.glob(os.path.join(data_dir, "testing", "*img.nii.gz"))) #KEN Start with just one image but later we can allow them to specificy multiple images
    test_files = [{"image": img} for img in zip(images)] #KEN Remove test and validation it's now an input and output

    # set some important parameters comparable to training #KEN see if these can go in the parameters file
    roi_size = (112,112)
    spatial_window_size = (112,112,1)
    spatial_window_batch_size = 1
    amount_of_labels = 9
    inference_iteration = 500000 #KEN We don't need inference_iteration any more
    pix_dim  = (1,1,-1)
    model_save_best = "best_metric_model_UNET_Lumbar_spine.pth"
    model_continue_training = f'model_UNET_Lumbar_spine_iteration_{inference_iteration}.pth'

    #set seed for reproducibility (identical to training part)
    set_determinism(seed=0) #Ken Don't belive this is necessary for inference

    #create transforms identical to training part, but here we don't specifiy the label key #Ken maybe call this inference instead of validation_original
    validation_original = Compose([
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
    validation_original_dataset = Dataset(
    data=test_files, transform=validation_original,
    )

    validation_original_loader = DataLoader(
    validation_original_dataset , batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    #device config #KEN probably want to keep
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #set post transforms #KEN can probably go into parameters file
    post_transforms = Compose([
    Invertd(
        keys="pred",
        transform=validation_original,
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
    KeepLargestConnectedComponentd(keys="pred", applied_labels=[1,2,3,4,5,6,7,8,9]),, #Ken Not all Models will have 1-9 labels
    SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=save_dir, output_dtype=('int16'), separate_folder = False, resample=False)
    ])

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
    model.load_state_dict(torch.load(
    os.path.join(root_dir,'Models',"lumbar_spine_UNET_per_iteration",model_continue_training )))
    model.eval()

    #Run inference on the image and images.

    #Inference part
    with torch.no_grad():
    for i, test_data in enumerate(validation_original_loader): #Ken call test_data input_images or inputs
        val_inputs = test_data["image"].to(device)
        axial_inferer = SliceInferer(roi_size=roi_size, sw_batch_size=spatial_window_batch_size, spatial_dim=2)
        test_data["pred"] = axial_inferer(val_inputs, model)
        val_data = [post_transforms(i) for i in decollate_batch(test_data)]




    """Optional"""
    #Allow list of filenames separated by spaces
    #Make sure relative and absolute paths work
    

if __name__ == "__main__":
    main()
    