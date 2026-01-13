---
title: Registration
parent: Key features
description: "MuscleMap's mm_register_to_template performs anatomical registration of MRI and CT scans to a standard template, enabling spatial normalisation, group comparisons, and voxelwise muscle analysis"
grand_parent: User section
nav_order: 3
permalink: /muscle-registration/
---


## Template-based muscle registration for voxel-wise muscle analysis

`mm_register_to_template` registers an input image and its segmentation to a predefined MuscleMap anatomical template.  
This enables standardized voxel-wise analyses, such as spatial parametric mapping of intramuscular fat, across subjects.

This step is typically performed after segmentation with `mm_segment`.

---

### 1. Basic usage

```bash
mm_register_to_template -i image.nii.gz -s image_dseg.nii.gz -r abdomen
```

This command:

- loads the input image (`-i`)
- loads the corresponding segmentation (`-s`)
- registers the image and each label to the selected regional template (`-r`)
- outputs warped images, segmentations, and transformation fields

<div class="callout callout-warning">
  <strong>Warning</strong><br>
  The segmentation must match the input image dimensions and orientation.  
  Always use the segmentation produced for that exact image.
</div>

---

### 2. Requirements

### 2.1 Spinal Cord Toolbox

This tool depends on Spinal Cord Toolbox (SCT) and expects:

- SCT version 6.5

<div class="callout callout-warning">
  <strong>Warning</strong><br>
  MuscleMap’s template registration workflow is developed and tested with  
  <strong>Spinal Cord Toolbox 6.5</strong>.  
  Other versions may not behave identically.
</div>

The SCT installation directory must be available via:

```bash
export SCT_DIR=/path/to/spinalcordtoolbox
```

---

### 3. How the registration works

For each label in the segmentation (each muscle or structure with label > 0), the following steps are performed:

1. Label extraction
   A binary mask is created for the current label.

2. Centerline estimation
   A slice-wise center-of-mass is computed to estimate a label-specific centerline.

3. Initial affine alignment
   Three landmark points (≈10%, 50%, 90% along the centerline) are used to compute a constrained affine transform, which is applied to:
   - the input image (linear interpolation)
   - the label mask (nearest-neighbour interpolation)

4. Nonlinear registration to the template
   Using `sct_register_multimodal`, the affine-initialized image and label are nonlinearly registered to the template.

5. Warp concatenation and application 
   Final transformation fields are concatenated and applied to generate outputs in template space.

---

### 4. Outputs

For each label, the following files are generated (naming simplified):

- `*_dseg_label-<L>.nii.gz` — binary label mask in native space  
- `*_label-<L>_affine.nii.gz` — image after initial affine alignment  
- `*_dseg_label-<L>_affine.nii.gz` — label after affine alignment  
- `warp_*_affine2<region>_template.nii.gz` — nonlinear warp field  
- `warp_*2<region>_template.nii.gz` — concatenated warp (native → template)  
- `*_label-<L>2<region>_template.nii.gz` — image in template space  
- `*_dseg_label-<L>2<region>_template.nii.gz` — label in template space

---

### 5. Quality control

Always visually inspect registration quality:

```bash
fsleyes template.nii.gz image_label-<L>2abdomen_template.nii.gz &
```

Check for:

- correct anatomical alignment  
- absence of flips or large shifts  
- plausible deformation of each label

<div class="callout callout-warning">
  <strong>Warning</strong><br>
  Registration may fail for very small or noisy labels.  
  Inspect representative labels before group analyses.
</div>

---

### 6. Voxel-wise correlation analysis (Randomise example)

After registration, template-space metric maps can be analysed voxel-wise.

### 6.1 Merge subjects into a 4D file

```bash
fslmerge -t all_subjects_metric_4D.nii.gz sub-01_metric.nii.gz sub-02_metric.nii.gz
```

### 6.2 Create a template-space mask

```bash
fslmaths mask.nii.gz -bin mask_bin.nii.gz
```

### 6.3 Run Randomise

```bash
randomise   -i all_subjects_metric_4D.nii.gz   -o stats_metric_corr   -m mask_bin.nii.gz   -d design.mat   -t design.con   -n 5000   -T
```

<div class="callout callout-note">
  <strong>Note</strong><br>
  Intermediate reflects a voxel with intermediate voxel signal not clearly corresponding to either fat or muscle.
</div>

<a class="mm-btn mm-btn-ghost mm-btn-back"
   href="{{ '/' | relative_url }}">
  ← Back to MuscleMap overview
</a>
