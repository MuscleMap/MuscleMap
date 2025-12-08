---
title: MuscleMap Toolbox
nav_order: 1
---

<p align="center">
  <img src="logo.png" alt="MuscleMap Logo" width="220" />
</p>

# MuscleMap Toolbox

MuscleMap is an open-source toolbox for whole-body muscle segmentation and analysis in large-scale imaging datasets.

It provides command line tools and a graphical user interface to:
- perform whole-body muscle and bone segmentation  
- extract quantitative metrics (volume, fat fraction, etc.)  
- visualise and quality-check segmentations  

---

## Get started

Use one of the following entry points:

- **Installation** – set up the MuscleMap environment on your machine  
- **Usage** – learn how to run the command line tools and the GUI  
- **(Later) Data structure** – how to organise your data for batch processing  

You can navigate using the menu on the left, or use the search bar at the top.

---

## Key components

### Command line tools

- `mm_segment`  
  Run whole-body segmentation on a single image or a batch of images.

- `mm_extract_metrics`  
  Compute region-wise metrics (e.g. GM/WM/marrow volumes, fat fraction).

- `mm_gui`  
  Open the graphical user interface for interactive inspection and QA.

---

## Who is this toolbox for?

MuscleMap is designed for:

- researchers working with whole-body MRI or CT  
- clinicians interested in quantitative muscle assessment  
- data scientists who want a reproducible segmentation pipeline  

If you’re new here, start with **Installation**, then follow the **Usage → Quick examples** section.
