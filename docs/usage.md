---
title: Usage
nav_order: 4
---


# Usage

MuscleMap provides three command-line tools: `mm_segment` for generating whole-body muscle and bone segmentations, `mm_extract_metrics` for computing region-wise quantitative metrics, and `mm_gui` for visual inspection and quality control. Below is a complete, continuous explanation of all three tools in one block.

A video demonstration is provided below.

<div style="margin: 1.5rem 0;">
  <iframe width="100%" height="400"
    src="https://www.youtube.com/embed/utlUVdvy6WI"
    title="MuscleMap Demonstration"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>


`mm_segment` is used to generate full-body anatomical segmentations from whole-body MRI scans. You run it by supplying an input image (`-i`) and an output file path (`-o`). A minimal example is:

mm_segment -i image.nii.gz -o image_dseg.nii.gz


You can specify the device using `--device cpu` or `--device cuda` if a GPU is available. For example:

mm_segment -i subject01_mri.nii.gz -o subject01_dseg.nii.gz --device cuda

The tool outputs a NIfTI segmentation file containing all labelled muscle and bone structures. Optional parameters include `--model` to select a specific checkpoint, `--patch-size` to override inference patch shapes, and `--min-size` to automatically pad smaller volumes. The output typically includes `*_dseg.nii.gz` plus an optional log file.

Once segmentation is complete, you can use `mm_extract_metrics` to compute quantitative measurements from the MRI and segmentation. This includes total and regional muscle volumes, fat fraction estimates using Gaussian Mixture Models (GMM) or thresholding, and intensity statistics for any region or group of regions. A typical command looks like:

mm_extract_metrics -m gmm -r wholebody -i image.nii.gz -s image_dseg.nii.gz -o metrics.csv

Here, `-m gmm` selects the fat-fraction method, `-r wholebody` selects the region grouping, `-i` and `-s` provide the MRI and segmentation, and `-o` defines the output CSV file. You can also specify `-c` to set the number of GMM components (commonly 3). Another example using thresholding is:

mm_extract_metrics -m threshold -r legs -i mri.nii.gz -s dseg.nii.gz -o legs_metrics.csv


The output is a CSV file listing region labels, volumes, fat-fraction measures, and summary statistics.

For visualisation and quality control, MuscleMap includes `mm_gui`, a graphical interface that lets you load MRI images and segmentation files interactively. You launch it simply with:



Inside the GUI, you can load an image, add its segmentation overlay, scroll through axial, coronal, or sagittal slices, adjust opacity, toggle muscle and bone regions, and save slices for documentation. This is typically used after running `mm_segment` to verify segmentation quality before computing metrics or running batch pipelines.

Together, these three tools form the standard workflow:  
1. Segment the MRI using `mm_segment`.  
2. Verify results using `mm_gui`.  
3. Compute quantitative outputs using `mm_extract_metrics`.  



