---
title: mm_extract_metrics
parent: MuscleMap commands
grand_parent: User section
nav_order: 2
permalink: /mm_extract_metrics/
---

# `mm_extract_metrics`

`mm_extract_metrics` is the MuscleMap command-line tool used to compute **quantitative muscle metrics** from MRI or CT images, using the segmentation produced by `mm_segment`.  
It supports Gaussian Mixture Models (GMM), Otsu thresholding, HU‑based fat quantification for CT, region‑based summaries, and customizable output formats.

This page explains:

- how `mm_extract_metrics` works  
- all key command-line options  
- recommended workflows  
- output files and their interpretation  
- troubleshooting guidance  

> **Tip:** For the most up-to-date options in your installed version, run:
> ```bash
> mm_extract_metrics -h
> ```

---

# 1. Basic usage

After generating a segmentation with `mm_segment`, run:

```bash
mm_extract_metrics -m gmm -r wholebody -i image.nii.gz -s image_dseg.nii.gz -c 3
```

This command:

- loads the image (`-i`)  
- loads the segmentation (`-s`)  
- applies the chosen metric method (`-m`)  
- computes fat/water composition or tissue-specific statistics  
- outputs CSV + NIfTI metric maps (depending on options)

---

# 2. Required inputs

## 2.1 `-i` — input image

The MRI/CT image from which metrics are extracted:

```bash
mm_extract_metrics -i sub-01_T2w.nii.gz
```

## 2.2 `-s` — segmentation labelmap

The labelmap produced by `mm_segment`:

```bash
mm_extract_metrics -s sub-01_dseg.nii.gz
```

The segmentation must contain the same dimensions and orientation as the input image.

> [!WARNING]
> If the segmentation does not match the input image dimensions, metrics cannot be computed.  
> Ensure you use the segmentation produced for that exact image.

---

# 3. Metric computation methods (`-m`)

The `-m` flag determines **how fat fraction / composition metrics are computed**.

Supported values:

### **1. `gmm` — Gaussian Mixture Model (MRI)**  
Uses a Gaussian Mixture Model to separate tissue intensities into classes (e.g., fat vs. muscle).

```bash
mm_extract_metrics -m gmm -i img.nii.gz -s img_dseg.nii.gz -c 3
```

Use this for:

- T1-weighted MRI  
- T2-weighted MRI  
- Dixon MRI (if not using raw water/fat channels)

### **2. `otsu` — Otsu thresholding (MRI)**  
Automatically selects intensity thresholds to separate tissues.

```bash
mm_extract_metrics -m otsu -i img.nii.gz -s img_dseg.nii.gz
```

Useful for quick processing or low-contrast MRI.

### **3. `hu` — Hounsfield Unit metrics (CT)**  
For CT images, muscle density is derived from Hounsfield Units.

```bash
mm_extract_metrics -m hu -i ct_img.nii.gz -s ct_dseg.nii.gz
```

Outputs metrics such as:

- mean HU per muscle  
- low-attenuation muscle area  
- voxel-level HU maps  

> [!TIP]
> Use `hu` metrics only for CT. MRI intensities do **not** reflect physical density.

---

# 4. Region selection (`-r`)

Choose which muscle regions to extract metrics for.

Example:

```bash
mm_extract_metrics -r wholebody
```

Common values include:

- `wholebody`  
- `abdomen`  
- `pelvis`  
- `thigh`  
- `leg`  

Regions correspond to MuscleMap's anatomical label groups.

> [!NOTE]
> Region definitions are based on the MuscleMap atlas and segmentation model.

---

# 5. Number of clusters (`-c`)

Used only with **GMM**.

Example for 3 clusters:

```bash
mm_extract_metrics -c 3
```

Typical choice:

- **2 clusters:** fat vs. muscle  
- **3 clusters:** fat, muscle, intermediate tissue  

---

# 6. Output files

`mm_extract_metrics` typically produces:

### **1. CSV file with summary statistics**
Contains per-muscle metrics such as:

- muscle volume  
- fat fraction  
- HU mean (for CT)  
- GMM cluster proportions  
- Otsu thresholding values  

### **2. Voxel-wise metric maps (optional depending on method)**

Examples:

- GMM class map  
- Otsu threshold map  
- HU-based fat fraction map (CT)

### **3. Optional NIfTI files for debugging or visualization**

---

# 7. Example workflows

## 7.1 MRI (T1/T2/Dixon) using GMM

```bash
mm_segment -i sub-01_T2w.nii.gz
mm_extract_metrics -m gmm -i sub-01_T2w.nii.gz -s sub-01_T2w_dseg.nii.gz -r wholebody -c 3
```

## 7.2 MRI using Otsu (fast)

```bash
mm_extract_metrics -m otsu -i img.nii.gz -s img_dseg.nii.gz -r pelvis
```

## 7.3 CT muscle density (HU)

```bash
mm_extract_metrics -m hu -i ct_img.nii.gz -s ct_dseg.nii.gz -r abdomen
```

---

# 8. Integration with `mm_gui`

If you prefer a GUI:

```bash
mm_gui
```

In the GUI you may:

- run `mm_segment`  
- run `mm_extract_metrics`  
- automatically chain them (segmentation → metrics → summary)

---

# 9. Best practices & troubleshooting

> [!IMPORTANT]
> Always **visually inspect both segmentation and metric outputs** before analysis.

### Common issues:

#### **Segmentation and image do not align**
- Ensure they come from the same subject/session.
- Ensure preprocessing did not change one but not the other.

#### **GMM produces incorrect clusters**
- Try a different number of clusters (`-c 2` or `-c 3`).
- Consider using Otsu if MRI contrast is low.

#### **CT metrics look wrong**
- Confirm method is set to `hu`.
- Ensure units are in **Hounsfield Units**, not rescaled.

#### **Output missing?**
Run with verbose logs:

```bash
mm_extract_metrics -v
```

---

# 10. Summary

`mm_extract_metrics` is the quantitative analysis backbone of MuscleMap:

- accepts MRI or CT images + segmentations  
- computes fat fraction, HU metrics, tissue composition  
- supports GMM, Otsu, HU-based methods  
- outputs CSV and optional maps  
- integrates with `mm_gui` for streamlined workflows  

It is designed for **reproducible, large-scale muscle phenotyping**.

---
