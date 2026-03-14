---
title: Neurodesk Webapps
parent: Key features
description:
grand_parent: User section
nav_order: 5
permalink: /neurodesk/
---

## MuscleMap Neurodesk Web Application

MuscleMap is also available through a Neurodesk-powered web application that allows fully automated whole-body muscle segmentation directly from your browser. This interface is designed for researchers, clinicians, and imaging scientists who want to easily run MuscleMap without installing any software.

The Neurodesk interface provides a streamlined workflow: simply upload your imaging data, run the segmentation, and visualise the results.- For more information about NeuroDesk, click [here](https://neurodesk.org/). 

<div class="mm-card mm-card-with-media mm-card-media-wide">
  <div class="mm-card-media">

<video autoplay loop muted playsinline>
  <source src="{{ '/assets/images/video_neurodesk.mp4' | relative_url }}" type="video/mp4">
</video>

  </div>
  <div class="mm-card-text">
    <div class="mm-card-actions">
      <a class="mm-btn mm-btn-primary"
           href="https://musclemap.neurodesk.org/"
           target="_blank" rel="noopener">
          Open MuscleMap Neurodesk
      </a>
    </div>
  </div>
</div>

---

### Why use MuscleMap Neurodesk?

MuscleMap Neurodesk provides the easiest way to run MuscleMap segmentation without any local installation or technical setup.

- No software installation required  
- Runs directly in your browser  
- Simply **drag and drop images** (NIfTI or DICOM) into the web interface  
- Click **Run segmentation** to start the automated analysis  
- Automated whole-body muscle segmentation  
- Interactive visualisation of segmentation results  

All processing runs inside the Neurodesk environment while your data **remains on your local workstation**, ensuring that imaging data never leaves your system. This makes the platform convenient while maintaining data security.

---

### Using MuscleMap Neurodesk

### 1. Open the Neurodesk application

Go to:  
https://musclemap.neurodesk.org/

---

### 2. Upload imaging data

You can upload data by:

- Dragging **NIfTI (`.nii`, `.nii.gz`)** files into the interface  
- Dragging **DICOM folders** directly into the application

The system automatically detects the image type and prepares it for segmentation.

---

### 3. Run MuscleMap segmentation

1. Select the uploaded image
2. Click:  
   **Run segmentation**

The model will automatically:

- Preprocess the image
- Run the MuscleMap segmentation model
- Generate labelled muscle segmentations
- Display the results in the viewer

---

### 4. Visualise and export results

After segmentation:

- The labelled segmentation appears automatically
- You can inspect the result using the integrated viewer
- Segmentation outputs can be exported for further quantitative analysis

---

## Reporting issues and requesting features

If you encounter an error, unexpected behaviour, or have a feature request, please open an issue on GitHub.

<a href="https://github.com/MuscleMap/MuscleMap/issues"
   class="mm-btn mm-btn-ghost"
   target="_blank" rel="noopener">
  Report an issue
</a>

Providing the following information helps us resolve issues faster:

- Browser and operating system
- Error message or screenshot
- Example data (if possible)

---

## Citation

If you use MuscleMap for academic or scientific work, please cite the MuscleMap consortium publications listed in the acknowledgements.

---

**MuscleMap** – Open-source, community-supported whole-body muscle quantification.

<a class="mm-btn mm-btn-ghost mm-btn-back"
   href="{{ '/' | relative_url }}">
  ← Back to MuscleMap overview
</a>