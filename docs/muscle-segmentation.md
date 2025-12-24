---
title: Segmentation
parent: Key features
description: "MuscleMap's mm_segment performs automatic whole-body muscle and bone segmentation on MRI and CT using deep learning. Supports 89 muscles, intramuscular fat analysis, and quantitative muscle metrics."
grand_parent: User section
nav_order: 1
permalink: /muscle-segmentation/
---

## Automated whole-body muscle segmentation

The segmentation is performed using a deep learning–based, contrast-agnostic whole-body model trained for robust muscle and bone segmentation across MRI and CT modalities.

This page explains:

- how to run segmentation module: `mm_segment`
- all available options and flags
- whole-body vs. regional models
- recommended workflows
- troubleshooting tips

> **Tip:** For the most up-to-date list of options in your installation, run:
>
> ```bash
> mm_segment --help
> ```

---

### 1. Basic usage

### 1.1 Activate the MuscleMap environment

```bash
conda activate MuscleMap
```

### 1.2 Run `mm_segment` on a single NIfTI image

First, navigate to the directory containing your input image (or provide the full path to the file):

```bash
cd /path/to/your/data
```

Then run:
```bash
mm_segment -i image.nii.gz
```

This command:

- loads the input image  
- applies the default whole-body segmentation model
- writes a segmentation labelmap (NIfTI) next to the input file  
- prints output paths and logs to the terminal  

<div class="callout callout-warning">
  <strong>Warning</strong><br>
  Running <code>mm_segment</code> from the command line works as expected when MuscleMap is installed in <strong>editable mode</strong> (<code>pip install -e .</code>).<br>
  If the package was installed using <code>pip install .</code>, it may be preferable to run the script directly using <code>python mm_segment.py</code>.
</div>


---

### 2. Input requirements

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

### 3. Output

`mm_segment` generates deep learning–based whole-body muscle segmentation outputs, including:

- a **segmentation labelmap** (NIfTI) with integer labels  
- an output file saved next to the input  
- terminal logs with paths and runtime info  

Visualise results in:

- [ITK-SNAP](https://www.itksnap.org/pmwiki/pmwiki.php)
- [3D Slicer](https://www.slicer.org/)
- [FSLeyes](https://open.oxcin.ox.ac.uk/pages/fslcourse/practicals/intro1/index.html)

---
### 4. Models and regions

### 4.1 Whole-body model (default)

```bash
mm_segment -i image.nii.gz
```

Segments:

- trunk muscles  
- pelvic muscles  
- thigh muscles
- leg muscles
- neck muscles  
- selected bones  

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

<div class="callout callout-warning">
  <strong>Warning</strong><br>
The legacy regional models are maintained for backward compatibility only.  
Active development and state-of-the-art performance are provided exclusively by the **whole-body model**, which achieves robust performance across all anatomical regions.
</div>

---

### 5. Command-line options

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

<div class="callout callout-note">
  <strong>Note</strong><br>
Lower sliding-window overlap increases inference speed but may reduce segmentation performance, particularly near tile boundaries.  
Higher overlap improves robustness at the cost of longer runtimes.
</div>

### 5.3 `-r` — region model

```bash
mm_segment -i image.nii.gz -r thigh
```

### 5.4 `-g` — GPU/CPU selection

Controls whether inference is performed on the GPU or CPU.

Example:

```bash
mm_segment -i image.nii.gz -g Y
```

<div class="callout callout-note">
  <strong>Note</strong><br>
By default, MuscleMap runs inference on the **GPU** when available.  
Use `-g N` to explicitly force CPU-based inference.
</div>


### 5.5 `-h` — help

```bash
mm_segment -h
```

---

### 6. Example workflows

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

See the [Muscle quantification and metric extraction](../muscle-quantification/) page for details on muscle volume and fat infiltration analysis.

### 7. Best practices & troubleshooting

- Always visually inspect segmentations  
- Poor image quality reduces accuracy  
- Record MuscleMap version and options in publications  

If something looks wrong:

- open a [GitHub issue](https://github.com/MuscleMap/MuscleMap/issues) with an example  

---
