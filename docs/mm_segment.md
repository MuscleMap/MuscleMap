---
title: mm_segment
parent: MuscleMap commands
description: "MuscleMap's mm_segment performs automatic whole-body muscle and bone segmentation on MRI and CT using deep learning. Supports 89 muscles, intramuscular fat analysis, and quantitative muscle metrics."
grand_parent: User section
nav_order: 1
permalink: /mm_segment/
---

MuscleMap provides automatic whole-body muscle segmentation for MRI and CT, using a deep learning model that identifies 89 muscles and bones. It supports muscle quantification, intramuscular fat measurement, and body composition analysis.

# `mm_segment`

`mm_segment` is the command-line tool in the MuscleMap toolbox for automatic segmentation of muscles and selected bones from axial MRI or CT images.  It uses a contrast-agnostic whole-body model that segments **89 muscles and bones** by default.

This page explains:

- how to run `mm_segment`
- all available options and flags
- whole-body vs. regional models
- recommended workflows
- troubleshooting tips

> **Tip:** For the most up-to-date list of options in your installation, run:
>
> ```bash
> mm_segment -h
> ```

---

## 1. Basic usage

### 1.1 Activate the MuscleMap environment

```bash
conda activate MuscleMap
```

### 1.2 Run `mm_segment` on a single NIfTI image

```bash
mm_segment -i image.nii.gz
```

This command:

- loads the input image  
- applies the default **whole-body segmentation model**  
- writes a segmentation labelmap (NIfTI) next to the input file  
- prints output paths and logs to the terminal  

---

## 2. Input requirements

### 2.1 Supported image types

- **Modality:** Axial MRI of any contrast (T1w, T2w, Dixon water/fat/in-phase) or CT  
- **Format:** NIfTI (`.nii`, `.nii.gz`)  
- **Orientation:** Axial orientation recommended  

MuscleMap recommends storing data in **BIDS format** with **JSON sidecars** from `dcm2niix`.

### 2.2 Processing single images vs. datasets

`mm_segment` processes **one image at a time**.

For multiple images:

- loop through images in shell/Python  
- keep consistent naming using BIDS  

---

## 3. Output

`mm_segment` generates:

- a **segmentation labelmap** (NIfTI) with integer labels  
- an output file saved next to the input  
- terminal logs with paths and runtime info  

Visualise results in:

- ITK-SNAP  
- 3D Slicer  
- FSLeyes  

---

## 4. Models and regions

### 4.1 Whole-body model (default)

```bash
mm_segment -i image.nii.gz
```

Segments:

- trunk muscles  
- pelvic muscles  
- thigh muscles  
- selected bones  
- many smaller groups (89 structures total)

### 4.2 Legacy regional models

Available:

- `abdomen`
- `pelvis`
- `thigh`
- `leg`

Example:

```bash
mm_segment -i image.nii.gz -r abdomen
```

---

## 5. Command-line options

### 5.1 `-i` — input image (required)

```bash
mm_segment -i /path/to/image.nii.gz
```

### 5.2 `-s` — sliding-window overlap

Controls tile overlap.

Example:

```bash
mm_segment -i image.nii.gz -s 50
```

### 5.3 `-r` — region model

```bash
mm_segment -i image.nii.gz -r thigh
```

### 5.4 `-g` — GPU/CPU selection

```bash
mm_segment -i image.nii.gz -g 0
```

### 5.5 `-h` — help

```bash
mm_segment -h
```

---

## 6. Example workflows

### 6.1 Whole-body segmentation

```bash
mm_segment -i sub-01_water.nii.gz
```

### 6.2 Faster inference

```bash
mm_segment -i sub-02_T2w.nii.gz -s 50
```

### 6.3 Using a regional model

```bash
mm_segment -i sub-03_T2w.nii.gz -r abdomen
```

### 6.4 Full pipeline with metrics

```bash
mm_segment -i image.nii.gz
mm_extract_metrics -m gmm -r wholebody -i image.nii.gz -s image_dseg.nii.gz -c 3
```

---

## 7. Best practices & troubleshooting

- Always visually inspect segmentations  
- Poor image quality reduces accuracy  
- Record MuscleMap version and options in publications  

If something looks wrong:

- verify NIfTI orientation  
- reduce overlap (`-s 50`)  
- check GPU installation  
- open a GitHub issue with an example  

---
