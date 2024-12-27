"""
--------------------------------------------------------------------------------
Train a Muscle Segmentation Model Using MONAI
--------------------------------------------------------------------------------
This script trains a muscle segmentation model based on user-defined parameters
in a JSON configuration file. It uses MONAI for data processing and augmentation,
PyTorch for deep learning, and logs all training events to both the console
and a file named `training.log`.

Major Steps:
  1) Parse command-line arguments and load the JSON configuration.
  2) Configure logging to output to console and `training.log`.
  3) Load training data (using a BIDS-like folder structure).
      - The labels are expected to end with .dseg.nii.gz
      - The images are expected to end with {sequence}.nii.gz (for example, fat.nii.gz)
  4) Create MONAI transforms for data augmentation, then build a CacheDataset.
  5) Initialize the UNet-based model (or another network via `initialize_model`).
  6) Set up the training loop:
     - Move data to GPU when available
     - Forward pass
     - Compute DiceCELoss
     - Backward pass with AMP scaling
     - Save checkpoints periodically
     - Log training progress (loss, step, LR)
  7) Complete training once `max_iterations` is reached.

To run:
  python train_script.py --config ./training_configuration.json
"""
import re
import os
import sys
import collections
import json
import argparse
import logging
import torch
from torch.optim import SGD
from monai.losses import DiceCELoss
from monai.utils import set_determinism
from monai.data import CacheDataset, ThreadDataLoader

try:
    # Attempt to import as if it is a part of a package
    from .mm_util import (
        get_transforms, initialize_model, poly_lr_scheduler, get_BIDS_data_files
    )
except ImportError:
    # Fallback to direct import if run as a standalone script
    from mm_util import (
        get_transforms, initialize_model, poly_lr_scheduler, get_BIDS_data_files
    )

# For reproducibility in data augmentation and some GPU/CUDA ops
set_determinism(seed=0)


def get_parser():
    """
    Creates and returns the ArgumentParser responsible for:
      --config : path to the JSON configuration file
    """
    parser = argparse.ArgumentParser(description="Train a muscle segmentation model")
    parser.add_argument(
        '--config',
        type=str,
        required=False,
        default="./training_configuration.json",
        help=(
            "Path to the configuration file containing hyperparameters, data paths, "
            "and training settings. Default is ./training_configuration.json"
        ),
    )
    return parser

def main():
    """
    Main entry point for the muscle segmentation training script.
    It:
      1) Configures Python's logging module (both file and console).
      2) Loads and parses the JSON configuration.
      3) Builds and prepares the MONAI datasets/dataloaders.
      4) Initializes the model and optimizer.
      5) Runs the training loop until reaching max_iterations.
      6) Logs key training info (loss, step, LR) and saves checkpoints.
    """
    # --------------------------------------------------------------------------
    # 1) Configure Logging
    # --------------------------------------------------------------------------
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # Overall logger level
    
    # Log to file (overwrites file if it exists), with timestamps and severity
    file_handler = logging.FileHandler('training.log', mode='w')
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Log to console (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_format)
    
    # Attach both handlers to this logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Command line arguments received: {sys.argv}")

    # --------------------------------------------------------------------------
    # 2) Parse Command-Line / Configuration
    # --------------------------------------------------------------------------
    parser = get_parser()
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    logger.info(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Extract training parameters from the config
    data_dir = config['data_dir']
    model_dir = config['model_dir']
    spatial_window_size = tuple(config['spatial_window_size'])
    spatial_window_batch_size = config['spatial_window_batch_size']
    batch_size_training = config['batch_size_training']
    amount_of_labels = config['amount_of_labels']
    pix_dim = tuple(config['pix_dim'])
    max_iterations = config['max_iterations']
    eval_num = config['eval_num']
    load_checkpoint_flag = config['load_checkpoint_flag']
    starting_iteration = config['starting_iteration']
    initial_learning_rate = config['initial_learning_rate']

    # Log the hyperparameters and file paths
    logger.info("Training Parameters:")
    logger.info(f"  Data Directory: {data_dir}")
    logger.info(f"  Model Directory: {model_dir}")
    logger.info(f"  Spatial Window Size: {spatial_window_size}")
    logger.info(f"  Batch Size (Window): {spatial_window_batch_size}")
    logger.info(f"  Batch Size (Training): {batch_size_training}")
    logger.info(f"  Number of Labels (Number of Muscles + 1): {amount_of_labels}")
    logger.info(f"  Pixel Dimensions: {pix_dim}")
    logger.info(f"  Stop training at iteration: {max_iterations}")
    logger.info(f"  Evaluation / Checkpoint Frequency: {eval_num}")
    logger.info(f"  Load Checkpoint Flag: {load_checkpoint_flag}")
    logger.info(f"  Starting Iteration: {starting_iteration}")

    # --------------------------------------------------------------------------
    # 3) Basic Setup (GPU, etc.)
    # --------------------------------------------------------------------------

    torch.cuda.empty_cache()  # Just frees some GPU memory if available
    torch.backends.cuda.matmul.allow_tf32 = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        logger.info("I'm using the GPU.")
    else:
        logger.warning(
            "GPU not available. Check your CUDA / PyTorch compatibility here: "
            "https://pytorch.org/get-started/locally/"
        )

    # --------------------------------------------------------------------------
    # 4) Data Preparation
    #    - This is where we gather training files and build transformations.
    # --------------------------------------------------------------------------
    train_files = get_BIDS_data_files(data_dir)  # BIDS-structured data fetch
    
    #Log the total pairs and which sequences have included for validation
    logger.info(f"Total training pairs from BIDS structure: {len(train_files)}")
    
    # A dictionary to count how many times each sequence appears
    sequence_counts = collections.defaultdict(int)
    for pair in train_files:
        base_name = os.path.basename(pair["image"])
        sequence_full = re.sub(r"\.nii(\.gz)?$", "", base_name)
        sequence_only = sequence_full.rsplit('_', 1)[-1]
        sequence_counts[sequence_only] += 1
    for seq, count in sequence_counts.items():
        logger.info(f"Total {seq} pairs = {count}")
    
    # Define Data Augmentation  
    transforms = get_transforms(
        pix_dim,
        spatial_window_size,
        spatial_window_batch_size
    )
    
    # CacheDataset: caches loaded data in memory to speed up training
    train_ds = CacheDataset(
        data=train_files,
        transform=transforms,
        copy_cache=False,
        num_workers=0  # set to > 0 if you want parallel data loading
    )
    
    # ThreadDataLoader: a MONAI loader that can handle some I/O in separate threads
    train_loader = ThreadDataLoader(
        train_ds,
        batch_size=batch_size_training,
        shuffle=True,
        pin_memory=False
    )

    # --------------------------------------------------------------------------
    # 5) Model Initialization
    # --------------------------------------------------------------------------
    model = initialize_model(
        model_dir=model_dir,
        starting_iteration=starting_iteration,
        amount_of_labels=amount_of_labels,
        device=device,
        load_checkpoint=load_checkpoint_flag
    )

    # Set up loss function (Dice + CE) for multi-label segmentation
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    torch.backends.cudnn.benchmark = True  

    # Define SGD optimizer
    optimizer = SGD(
        model.parameters(),
        lr=initial_learning_rate,
        momentum=0.99,
        nesterov=True,
        weight_decay=1e-4
    )

    # Automatic Mixed Precision scaler
    scaler = torch.amp.GradScaler()

    # Global step tracks total iterations (across epochs, if any)
    global_step = starting_iteration

    logger.info("Starting training loop...")

    # --------------------------------------------------------------------------
    # 6) Training Loop
    #    - For each batch: forward pass, compute loss, backward pass, step.
    #    - Periodically save checkpoints and log progress.
    # --------------------------------------------------------------------------

    while global_step < max_iterations:
        # Put model in training mode 
        model.train()

        for step, batch in enumerate(train_loader):
            # Adjust the learning rate each iteration with a polynomial schedule
            current_lr = poly_lr_scheduler(global_step, max_iterations, initial_learning_rate)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            # Move data to GPU (if available)
            inputs = batch["image"].to(device=device, non_blocking=True).contiguous()
            labels = batch["label"].to(device=device, non_blocking=True).contiguous()

            # Reset gradients
            optimizer.zero_grad()

            # Mixed precision context
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            # Backprop using AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Checkpoint saving logic
            if (global_step % eval_num == 0) or (global_step == max_iterations):
                ckpt_name = f"model_iteration_{global_step}.pth"
                ckpt_path = os.path.join(model_dir, ckpt_name)
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_name}")

            global_step += 1

            # Log current iteration, loss, and LR
            logger.info(f"Step {global_step} / {max_iterations} | "
                        f"Loss = {loss.item():.4f} | "
                        f"LR = {current_lr:.4e}")

            # Stop if we reached the maximum number of iterations
            if global_step >= max_iterations:
                break

    logger.info("Training completed successfully!")

if __name__ == '__main__':
    main()
