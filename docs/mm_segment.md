---
title: mm_segment
parent: MuscleMap Commands
grand_parent: User section
nav_order: 1
---



`mm_segment` is used to generate full-body anatomical segmentations from whole-body MRI scans. You run it by supplying an input image (`-i`) and an output file path (`-o`). A minimal example is:

mm_segment -i image.nii.gz -o image_dseg.nii.gz

You can specify the device using `--device cpu` or `--device cuda` if a GPU is available. For example:

mm_segment -i subject01_mri.nii.gz -o subject01_dseg.nii.gz --device cuda

The tool outputs a NIfTI segmentation file containing all labelled muscle and bone structures. Optional parameters include `--model` to select a specific checkpoint, `--patch-size` to override inference patch shapes, and `--min-size` to automatically pad smaller volumes. The output typically includes `*_dseg.nii.gz` plus an optional log file.
