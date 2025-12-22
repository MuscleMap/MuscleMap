---
title: Graphical user interface
parent: Key features
grand_parent: User section
nav_order: 4
permalink: /graphical-user-interface/
---

# 3D Slicer Extension for Muscle Segmentation

MuscleMap is implemented as a dedicated 3D Slicer extension that enables fully automated whole-body muscle segmentation directly within the Slicer environment. This interface is designed for researchers, clinicians, and imaging scientists who want to apply MuscleMap models without writing code.

The GUI provides an end-to-end workflow: from loading CT or MRI data, installing dependencies, running segmentation, to interactive 3D visualisation of muscle labels.

<div class="mm-card mm-card-with-media">
  <div class="mm-card-media">
    <img src="{{ '/assets/images/MuscleMap_Slicer3D.png' | relative_url }}"
         alt="MuscleMap 3D Slicer Interface">
  </div>
</div>

---

## Why use the MuscleMap 3D Slicer extension?

- No programming required  
- Fully integrated into 3D Slicer  
- Supports CT and MRI
- Automated whole-body muscle segmentation
- Interactive 2D and 3D visualisation
- Export-ready segmentation labels for quantitative analysis

---
## Installing the MuscleMap extension in 3D Slicer

### Step 1 – Install 3D Slicer
Download and install the latest version of 3D Slicer (≥ 5.2) from:  
https://www.slicer.org

---
### Step 2 – Install the PyTorch (SlicerPyTorch) extension

MuscleMap requires PyTorch to run deep-learning models inside Slicer.

1. Open 3D Slicer
2. Go to:  
   View → Extensions Manager
3. Search for:  
   PyTorch or SlicerPyTorch
4. Install the extension
5. Restart 3D Slicer

---

### Step 3 – Install PyTorch inside Slicer

After restarting Slicer:

1. Go to:  
   View → Modules
2. Navigate to:  
   Utilities → PyTorch (SlicerPyTorch)
3. Click:  
   Install PyTorch
4. Choose:
   - **GPU version** (recommended if a compatible GPU is available)
   - or **CPU version** otherwise

Wait until the installation is completed successfully.

---

### Step 4 – Install the MuscleMap extension

1. Open Extensions Manager again
2. Search for:  
   MuscleMap
3. Install the extension
4. Restart 3D Slicer

After restarting, the module is available under the MuscleMap category.

---

## Using the MuscleMap GUI

### 1. Open the MuscleMap module
Go to:  
View → Modules → MuscleMap → MuscleMap Whole-Body Segmentation

---

### 2. Install MuscleMap dependencies (one-time step)

Inside the module:

1. Open Installing packages
2. Click:  
   Install MuscleMap dependencies

This installs MONAI, nibabel, pandas, and the MuscleMap toolbox inside Slicer’s Python environment.

---

### 3. Load an image volume

You can load data in two ways:

- Click Load volume from file…
- Or select an already loaded volume using Input volume

Supported formats include:
- `.nii`, `.nii.gz`

Both **CT and MRI** are supported.

---

### 4. Run MuscleMap segmentation

1. Select the input volume
2. (Optional) Enable **Force CPU** under *Advanced* if needed
3. Click:  
   Run MuscleMap segmentation

The model will automatically:
- Export the volume
- Run `mm_segment`
- Load the resulting segmentation
- Apply anatomical labels and colors
- Display the result in 2D and 3D

---

### 5. Visualise results in 3D

After segmentation:
- The output segmentation appears automatically
- Click Show 3D to enable 3D rendering
- Use standard Slicer tools for inspection, annotations or export

---

## Source code and development

The 3D Slicer extension is open-source and actively developed.

👉 **Slicer extension repository:**  
https://github.com/Eddowesselink/SlicerMuscleMap

<a href="https://github.com/Eddowesselink/SlicerMuscleMap" class="btn btn-primary" target="_blank">
  Open SlicerMuscleMap on GitHub
</a>

---

## Reporting issues and requesting features

If you encounter an error, unexpected behaviour, or have a feature request, please open an issue on GitHub.

👉 **MuscleMap issue tracker:**  
https://github.com/MuscleMap/MuscleMap/issues

<a href="https://github.com/MuscleMap/MuscleMap/issues" class="btn btn-outline" target="_blank">
  Report an issue
</a>

Providing the following information helps us resolve issues faster:
- 3D Slicer version
- Operating system
- CPU/GPU details
- Error message or log output
- Example data (if possible)

---

## Citation

If you use MuscleMap for academic or scientific work, please cite the MuscleMap consortium publications listed in the module acknowledgements.

---

**MuscleMap** – Open-source, community-supported whole-body muscle quantification.
