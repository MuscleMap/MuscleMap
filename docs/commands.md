---
title: MuscleMap Commands
nav_order: 2
parent: User section
has_children: true
---

# Usage

MuscleMap provides three command-line tools: `mm_segment` for generating whole-body muscle and bone segmentations, `mm_extract_metrics` for computing region-wise quantitative metrics, and `mm_gui` for visual inspection and quality control. Below is a complete, continuous explanation of all three tools in one block.

A video demonstration is provided below.

<div class="mm-video-container">
  <iframe
    src="https://www.youtube.com/embed/utlUVdvy6WI"
    title="MuscleMap Demonstration"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen
  ></iframe>
</div>



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



