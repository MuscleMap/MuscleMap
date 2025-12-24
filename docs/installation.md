---
title: Installation
nav_order: 1
parent: User section
permalink: /installation/
---

## Installation

This guide explains how to install MuscleMap and its dependencies for automated muscle segmentation and quantification from MRI and CT scans.

---

### Dependencies

- **Python:** 3.9.23  
- **Operating system:** Linux, macOS, or Windows  

The MuscleMap Toolbox works on **CPU and GPU**, but performance is substantially better with GPU acceleration.

---

### 1. Install Anaconda 

We recommend installing **Miniconda** or **Anaconda** through the following links:

- <a href="https://docs.conda.io/en/latest/miniconda.html" target="_blank" rel="noopener noreferrer">Miniconda</a>  
- <a href="https://www.anaconda.com/download" target="_blank" rel="noopener noreferrer">Anaconda</a>

---
### 2. Create and activate a conda environment

```bash
conda create --name MuscleMap python=3.9.23
conda activate MuscleMap
```

---
### 3. Download the MuscleMap repository

### Option A — Using Git

```bash
git clone https://github.com/MuscleMap/MuscleMap.git
cd MuscleMap
```

### Option B — Download ZIP

1. Open https://github.com/MuscleMap/MuscleMap  
2. Click the green **<> Code ▼** button  
3. Click **Download ZIP**  
4. Unzip the archive  
5. Navigate to the extracted folder:

```bash
cd MuscleMap
```
---

### 4. Install MuscleMap in editable mode

```bash
pip install -e .
```

This installs all required Python dependencies and registers the command-line tools:

- `mm_segment`
- `mm_extract_metrics`
- `mm_register_to_template`
- `mm_gui`

<div class="callout callout-note">
  <strong>Note</strong><br>
Installing MuscleMap includes a default CPU-only installation of PyTorch. If you want to use a GPU for faster inference, please proceed to Step 5.
</div>

---

### 5. (Optional) Install PyTorch with GPU support

If you plan to run MuscleMap **on CPU only**, you may skip this step.

To use a GPU, you need one of the following:

- **NVIDIA GPU** with a compatible CUDA runtime  
- **AMD GPU** with ROCm support  

### Step 5.1 — Check if CUDA is already available

Open a Python console:

```python
import torch
print("Is CUDA available?:", torch.cuda.is_available())
```

- `True` → CUDA is available and ready  
- `False` → continue with the steps below

---
### Step 5.2 — Check your system GPU runtime

In a terminal, run:

**NVIDIA (CUDA):**
```bash
nvidia-smi
```

**AMD (ROCm):**
```bash
rocm-smi
```

This tells you which CUDA or ROCm version your system supports.
---

### Step 5.3 — Install a compatible PyTorch version

Install **PyTorch 2.4.0** matching your system configuration.

We recommend using `pip` and following the official PyTorch instructions.

Find the Pytorch 2.4.0 installation <a href="https://pytorch.org/get-started/previous-versions/" target="_blank" rel="noopener noreferrer">here</a>  

Example (CUDA, adjust version as needed):

```bash
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

<div class="callout callout-note">
  <strong>Note</strong><br>
  Make sure the CUDA version of PyTorch matches the CUDA runtime reported by <code>nvidia-smi</code>. Also, we recommend to install Pytorch using the wheel. 
</div>

---

### 6. Verify the installation

Run:

```bash
mm_segment --help
mm_extract_metrics --help
mm_register_to_template --help
mm_gui --help
```

If these commands print a help message instead of an error, the installation was successful.

---