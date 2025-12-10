---
title: Change log 
nav_order: 1
parent: Developer section
---


# Changelog
All notable changes to **MuscleMap** are listed below.  
Source: GitHub Releases (https://github.com/MuscleMap/MuscleMap/releases)

---

## **v1.1** — *September 1*
- Trained a new whole-body segmentation model on a larger and more diverse dataset, improving generalizability.  
- Implemented foreground cropping to make inference more efficient.

---

## **v1.0** — *August 10*
- Added the **whole-body segmentation model (v1.0)**.  
- Updated `mm_segment` to better handle large images and improve segmentation efficiency.  
- Added several new options to `mm_segment`.  
- Cleaned and improved `mm_extract_metrics`.

---

## **v0.3 (Pre-release)** — *July 24*
- Introduced version numbers for trained models.  
- Preparation for the v1.0 release.

---

## **v0.2 (Pre-release)** — *February 11*
- Added `mm_register_to_template` for anatomical registration to template space.

---

## **v0.1 (Pre-release)** — *December 14*
Initial public pre-release of MuscleMap.  
Includes:
- `mm_segment` (muscle segmentation for abdomen, pelvis, thigh, leg)  
- `mm_extract_metrics` (muscle size & composition extraction)  
- `mm_gui` (graphical user interface)

---

## **v0.0 (Pre-release)** — *September 11*
- Initial repository version with early segmentation tools (abdomen-focused).

---
