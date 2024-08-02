# MuscleMap: An Open-Source, Community-Supported Consortium for Whole-Body Quantitative MRI of Muscle

## Planned MuscleMap Phases
\
<img align="left" width="15%" src="https://github.com/MuscleMap/MuscleMap/blob/main/logo.png">
1. Develop a standardized acquisition protocol for whole-body quantitative MRI of muscle for the most common MR manufacturers.

2. Generate an open-source large (n≥1,000) annotated multi-site, multi-racial, and multi-ethnic heterogenous whole-body muscle MRI dataset across the lifespan using the standardized acquisition protocol.

3. Create an open-source toolbox for the analysis of whole-body muscle morphometry and composition using the hetergenous whole-body muscle MRI dataset.
<br />

## MuscleMap Toolbox

### Dependencies

* Python 3.9.0

### Installation

1. Install python:
   * We recommend installing [Miniconda](https://docs.anaconda.com/miniconda) or [Anaconda](https://docs.anaconda.com/anaconda).

2. Create python environment:
    ~~~
    conda create --name MuscleMap python==3.9.0
    ~~~

3. Activate python environment:
    ~~~
    conda activate MuscleMap
    ~~~

4. Download MuscleMap repository:
    1. Using the git command line tool:
        ~~~
        git clone https://github.com/MuscleMap/MuscleMap
        ~~~
    
    2. From your browser:
    
        1. Open https://github.com/MuscleMap/MuscleMap in your browser

        2. Click the green **<> Code ▼** button

        3. Click **Download Zip**

        4. Unzip the MuscleMap repository

    3. Navigate to MuscleMap repository:

        ~~~
        cd ./MuscleMap
        ~~~

    4. Install python packages:
    
        ~~~
        pip install -r requirements.txt
        ~~~
   5. To use a GPU to speed up analyses, you will need a NVIDIA GPU and [CUDA](https://developer.nvidia.com/cuda-toolkit) installed.

### Usage

1. Navigate to MuscleMap repository scripts directory:

    ~~~
    cd ./MuscleMap/scripts
    ~~~

2. To run mm_segment:

    ~~~
    python mm_segment.py -i image.nii.gz -r abdomen -o output.nii.gz
    ~~~

    * mm_segment uses contrast agnostic segmentation models by default and only the body region needs to be specified. Users may specify another available model with -m.
    
    #### Regions
    * Abdomen
        * Left and right multifidus, erector spinae, psoas major, and quadratus lumborum
    
    *Regions in development: neck, shoulder, arm, forearm, thorax, pelvis, thigh, leg, and foot*

3. To run mm_extract_metrics:

    1. For T1w and T2w MRI:

        ~~~
        python mm_extract_metrics.py -m gmm -i image.nii.gz -s image_dlabel.nii.gz -c 3
        ~~~

    * Users may specify Gaussian mixture modeling (gmm) or kmeans clustering (kmeans) with -m.
    * Users may specifcy 2 or 3 components with -c.

    2. For Dixon fat-water MRI:

        ~~~
        python mm_extract_metrics.py -m dixon -f fat_image.nii.gz -w water_image.nii.gz -s image_dlabel.nii.gz
        ~~~

## Generic Acquisition Protocol

We are currently developing the genmeric acquisition protocol for whole-body quantitative MRI of Muscle. You can access the Google doc [here](https://docs.google.com/document/d/1q7AAnPEr7Rj5gb9d_mLrRnAiav1f32J-RPswvOPk5xE/edit?usp=sharing). To collaborate on the generic acquisition protocol, please contact [Kenneth Weber](mailto:kenweber@stanford.edu), [Eddo Wesselink](mailto:eddo_wesselink@msn.com), [James Elliott](mailto:james.elliott@sydney.edu.au), or [Marnee McKay](mailto:marnee.mckay@sydney.edu.au).

## Data Curation
We strongly recommend following the [Brain Imaging Data Structure (BIDS)](https://bids.neuroimaging.io/) specification for organizing your dataset. 

### Convert DICOM to BIDS

1. Convert images from [DICOM](https://www.dicomstandard.org/) format to [NIfTI](https://nifti.nimh.nih.gov/) format.
    * We recommend using [dcm2niix](https://github.com/rordenlab/dcm2niix) and working with compressed [NIfTI](https://nifti.nimh.nih.gov/) files (nii.gz)
    * Keep the [json](https://en.wikipedia.org/wiki/JSON) sidecar file, which contains imaging parameters

2. Rename the [NIfTI](https://nifti.nimh.nih.gov/) and [json](https://en.wikipedia.org/wiki/JSON) files and organize your dataset to follow the [BIDS](https://bids.neuroimaging.io/) specification.

3. Here is an example [BIDS](https://bids.neuroimaging.io/) directory structure:

    ```
    dataset
    ├── derivatives
    │   └── labels
    │       └── sub-example01
    │		└── sub-example02
    │           ├── ses-abdomen
    │           │   └── anat
    │           │       ├── sub-example02_ses-abdomen_T2w_label-muscle_dlabel.json
    │           │       └── sub-example02_ses-adomen_T2w_label-muscle_dlabel.nii.gz
    │           │       ├── sub-example02_ses-abdomen_water_label-muscle_dlabel.json
    │           │       └── sub-example02_ses-adomen_water_label-muscle_dlabel.nii.gz
    │           └── ses-neck
    │               └── anat
    │                   ├── sub-example02_ses-neck_water_label-muscle_dlabel.json
    │                   └── sub-example02_ses-neck_water_label-muscle_dlabel.nii.gz
    └── sourcedata
        └── participants.tsv
        └── sub-example01
        └── sub-example02
            ├── ses-abdomen
            │   ├── anat
            │   │   ├── sub-example02_ses-abdomen_fatfrac.json
            │   │   ├── sub-example02_ses-abdomen_fatfrac.nii.gz
            │   │   ├── sub-example02_ses-abdomen_fat.json
            │   │   ├── sub-example02_ses-abdomen_fat.nii.gz
            │   │   ├── sub-example02_ses-abdomen_inphase.json
            │   │   ├── sub-example02_ses-abdomen_inphase.nii.gz
            │   │   ├── sub-example02_ses-abdomen_outphase.json
            │   │   ├── sub-example02_ses-abdomen_outphase.nii.gz
            │   │   ├── sub-example02_ses-abdomen_R2star.json
            │   │   ├── sub-example02_ses-abdomen_R2star.nii.gz
            │   │   ├── sub-example02_ses-abdomen_T1w.json
            │   │   ├── sub-example02_ses-abdomen_T1w.nii.gz
            │   │   ├── sub-example02_ses-abdomen_T2w.json
            │   │   ├── sub-example02_ses-abdomen_T2w.nii.gz
            │   │   ├── sub-example02_ses-abdomen_water.json
            │   │   └── sub-example02_ses-abdomen_water.nii.gz
            │   └── dwi
            │       ├── sub-example02_ses-abdomen_dwi.bval
            │       ├── sub-example02_ses-abdomen_dwi.bvec
            │       ├── sub-example02_ses-abdomen_dwi.json
            │       └── sub-example02_ses-abdomen_dwi.nii.gz
            └── ses-neck
                └── anat
                    ├── sub-example02_ses-neck_fat.json
                    ├── sub-example02_ses-neck_fat.nii.gz
                    ├── sub-example02_ses-neck_fatfrac.json
                    ├── sub-example02_ses-neck_fatfrac.nii.gz
                    ├── sub-example02_ses-neck_inphase.json
                    ├── sub-example02_ses-neck_inphase.nii.gz
                    ├── sub-example02_ses-neck_outphase.json
                    ├── sub-example02_ses-neck_outphase.nii.gz
                    ├── sub-example02_ses-neck_R2star.json
                    ├── sub-example02_ses-neck_R2star.nii.gz
                    ├── sub-example02_ses-neck_T2w.json
                    ├── sub-example02_ses-neck_T2w.nii.gz
                    ├── sub-example02_ses-neck_water.json
                    └── sub-example02_ses-neck_water.nii.gz

    ```

    * sourcedata = contains participants.tsv, raw images, json sidecar files, and no other files
    * derivatives = contains segmentation images and any other derivatives
    * If you have a large dataset to convert, the [DICOM](https://www.dicomstandard.org/) to [BIDS](https://bids.neuroimaging.io/) conversion can be automated. If needed, feel free to reach out the MuscleMap developers for help automating the conversion.

## Citing MuscleMap

When using the MuscleMap Toolbox please cite the following publications.

## mm_segment

### Abdomen

* Wesselink EO, Elliott JM, Coppieters MW, Hancock MJ, Cronin B, Pool-Goudzwaard A, Weber II KA.Convolutional neural networks for the automatic segmentation of lumbar paraspinal muscles in people with low back pain. Sci Rep. 2022;12(1):13485. <https://doi.org/10.1038/s41598-022-16710-5>

* Wesselink EO, Pool-Goudzwaard A, De Leener B, Law CSW, Fenyo MB, Ello GM, Coppieters MW, Elliott JM, Mackey S, Weber KA 2nd. Investigating the associations between lumbar paraspinal muscle health and age, BMI, sex, physical activity, and back pain using an automated computer-vision model: a UK Biobank study. Spine J. 2024;24(7):1253-1266. <https://doi.org/10.1016/j.spinee.2024.02.013>

## mm_extract_metric

* Wesselink EO, Elliott JM, Pool-Goudzwaard A, Coppieters MW, Pevenage PP, Di Ieva A, Weber II KA. Quantifying lumbar paraspinal intramuscular fat: Accuracy and reliability of automated thresholding models. N Am Spine Soc J. 2024;17:100313. <https://doi.org/10.1016/j.xnsj.2024.100313>

## Publications

### 2024

* Wesselink EO, Pool-Goudzwaard A, De Leener B, Law CSW, Fenyo MB, Ello GM, Coppieters MW, Elliott JM, Mackey S, Weber KA 2nd. Investigating the associations between lumbar paraspinal muscle health and age, BMI, sex, physical activity, and back pain using an automated computer-vision model: a UK Biobank study. Spine J. 2024;24(7):1253-1266. <https://doi.org/10.1016/j.spinee.2024.02.013>

* Wesselink EO, Elliott JM, Pool-Goudzwaard A, Coppieters MW, Pevenage PP, Di Ieva A, Weber II KA. Quantifying lumbar paraspinal intramuscular fat: Accuracy and reliability of automated thresholding models. N Am Spine Soc J. 2024;17:100313. <https://doi.org/10.1016/j.xnsj.2024.100313>

* Perraton Z, Mosler AB, Lawrenson PR, Weber II K, Elliott JM, Wesselink EO, Crossley KM, Kemp JL, Stewart C, Girdwood M, King MG, Heerey JJ, Scholes MJ, Mentiplay BF, Semciw AI. The association between lateral hip muscle size/intramuscular fat infiltration and hip strength in active young adults with long standing hip/groin pain. Phys Ther Sport. 2024;65:95-101. <https://doi.org/10.1016/j.ptsp.2023.11.007>

### 2023

* Wesselink EO, Pool JJM, Mollema J, Weber KA 2nd, Elliott JM, Coppieters MW, Pool-Goudzwaard AL. Is fatty infiltration in paraspinal muscles reversible with exercise in people with low back pain? A systematic review. Eur Spine J. 2023;32(3):787-796. <https://doi.org/10.1007/s00586-022-07471-w>

### 2022

* Wesselink EO, Elliott JM, Coppieters MW, Hancock MJ, Cronin B, Pool-Goudzwaard A, Weber II KA.Convolutional neural networks for the automatic segmentation of lumbar paraspinal muscles in people with low back pain. Sci Rep. 2022;12(1):13485. <https://doi.org/10.1038/s41598-022-16710-5>

* Bodkin SG, Smith AC, Bergman BC, Huo D, Weber KA, Zarini S, Kahn D, Garfield A, Macias E, Harris-Love MO. Utilization of Mid-Thigh Magnetic Resonance Imaging to Predict Lean Body Mass and Knee Extensor Strength in Obese Adults. Front Rehabil Sci. 2022;3:808538. <https://doi.org/10.3389/fresc.2022.808538>

* Snodgrass SJ, Stanwell P, Weber KA, Shepherd S, Kennedy O, Thompson HJ, Elliott JM. Greater muscle volume and muscle fat infiltrate in the deep cervical spine extensor muscles (multifidus with semispinalis cervicis) in individuals with chronic idiopathic neck pain compared to age and sex-matched asymptomatic controls: a cross-sectional study. BMC Musculoskelet Disord. 2022;23(1):973. <https://doi.org/10.1186/s12891-022-05924-3>

* Franettovich Smith MM, Mendis MD, Weber KA 2nd, Elliott JM, Ho R, Wilkes MJ, Collins NJ. Improving the measurement of intrinsic foot muscle morphology and composition from high-field (7T) magnetic resonance imaging. J Biomech. 2022;140:111164. <https://doi.org/10.1016/j.jbiomech.2022.111164>

* Perraton Z, Lawrenson P, Mosler AB, Elliott JM, Weber KA 2nd, Flack NA, Cornwall J, Crawford RJ, Stewart C, Semciw AI. Towards defining muscular regions of interest from axial magnetic resonance imaging with anatomical cross-reference: a scoping review of lateral hip musculature. BMC Musculoskelet Disord. 2022;23(1):533. <https://doi.org/10.1186/s12891-022-05439-x>

### 2021

* Paliwal M, Weber KA 2nd, Smith AC, Elliott JM, Muhammad F, Dahdaleh NS, Bodurka J, Dhaher Y, Parrish TB, Mackey S, Smith ZA. Fatty infiltration in cervical flexors and extensors in patients with degenerative cervical myelopathy using a multi-muscle segmentation model. PLoS One. 2021;16(6):e0253863. <https://doi.org/10.1371/journal.pone.0253863>

* Weber KA 2nd, Abbott R, Bojilov V, Smith AC, Wasielewski M, Hastie TJ, Parrish TB, Mackey S, Elliott JM. Multi-muscle deep learning segmentation to automate the quantification of muscle fat infiltration in cervical spine conditions. Sci Rep. 2021;11(1):16567. <https://doi.org/10.1038/s41598-021-95972-x>

### 2020

* Elliott JM, Smith AC, Hoggarth MA, Albin SR, Weber KA 2nd, Haager M, Fundaun J, Wasielewski M, Courtney DM, Parrish TB. Muscle fat infiltration following whiplash: A computed tomography and magnetic resonance imaging comparison. PLoS One. 2020;15(6):e0234061. <https://doi.org/10.1371/journal.pone.0234061>

* Franettovich Smith MM, Collins NJ, Mellor R, Grimaldi A, Elliott J, Hoggarth M, Weber II KA, Vicenzino B. Foot exercise plus education versus wait and see for the treatment of plantar heel pain (FEET trial): a protocol for a feasibility study. J Foot Ankle Res. 2020;13(1):20. <https://doi.org/10.1186/s13047-020-00384-1>

### 2019

* Weber KA, Smith AC, Wasielewski M, Eghtesad K, Upadhyayula PA, Wintermark M, Hastie TJ, Parrish TB, Mackey S, Elliott JM. Deep Learning Convolutional Neural Networks for the Automatic Quantification of Muscle Fat Infiltration Following Whiplash Injury. Sci Rep. 2019;9(1):7973. <https://doi.org/10.1038/s41598-019-44416-8>

### 2017

* Smith AC, Weber KA, Parrish TB, Hornby TG, Tysseling VM, McPherson JG, Wasielewski M, Elliott JM. Ambulatory function in motor incomplete spinal cord injury: a magnetic resonance imaging study of spinal cord edema and lower extremity muscle morphometry. Spinal Cord. 2017;55(7):672-678. <https://doi.org/10.1038/sc.2017.18>
