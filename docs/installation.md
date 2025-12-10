---
title: Installation
nav_order: 1
parent: User section

---

# Installation

Below we describe a typical setup using `conda` and an editable install of MuscleMap.
Adapt paths and Python versions as needed.

## 1. Clone the repository

~~~bash
git clone https://github.com/MuscleMap/MuscleMap.git
cd MuscleMap
~~~

## 2. Create and activate a conda environment

~~~bash
conda create --name MuscleMap python=3.9.13
conda activate MuscleMap
~~~

## 3. Install MuscleMap in editable mode

~~~bash
pip install -e .
~~~

This will install all required Python dependencies and register the command line
entry points:

- `mm_segment`
- `mm_extract_metrics`
- `mm_extract_metrics_batch` (if available)
- `mm_gui`

## 4. Verify the installation

~~~bash
mm_segment --help
mm_extract_metrics --help
mm_gui --help
~~~

If these commands show a help message instead of an error, the installation is successful.