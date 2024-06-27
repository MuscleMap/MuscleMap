# MuscleMap: An Open-Source, Community-Supported Consortium for Whole-Body Quantitative MRI of Muscle

This repository includes the toolbox and generic acquisition protocol for the MuscleMap Consortium.

## Planned MuscleMap Phases
<img align="left" width="15%" src="https://github.com/MuscleMap/MuscleMap/blob/main/logo.png">

1. Develop a generic acquisition protocol for whole-body quantitative MRI of muscle for the most common MR manufacturers.
2. Generate an open-source large (n≥300) annotated multi-site, multi-racial, and multi-ethnic heterogenous whole-body Muscle MRI dataset across the lifespan using the generic acquisition protocol.
3. Create an open-source toolbox for the analysis of whole-body muscle morphometry and composition.

## MuscleMap Toolbox

### Dependencies

* Python 3.9.0


### Installation

1. Create python environement
    ~~~
    conda create --name MuscleMap python==3.9.0
    ~~~

2. Activate environement
    ~~~
    conda activate MuscleMap
    ~~~

3. Download MuscleMap repository
    1. Using the git command line tool:
        ~~~
        git clone https://github.com/MuscleMap/MuscleMap
        ~~~
    
    2. From your browser:
    
        1. Open https://github.com/MuscleMap/MuscleMap in your browser.

        2. Click green **<> Code ▼**

        3. Click **Download Zip**

        4. Unzip the zipped MuscleMap repository

    3. Navigate to MuscleMap repository

        ~~~
        cd ./MuscleMap
        ~~~

    4. Install python packages 
    
        ~~~
        pip install -r requirements.txt
        ~~~



### Usage

1. Navigate to MuscleMap repository scripts directory

    ~~~
    cd ./MuscleMap/scripts
    ~~~

2. mm_segment.py

    ~~~
    python mm_segment.py -i image.nii.gz -r abdomen -o output.nii.gz
    ~~~

3. mm_extract_metrics.py

    ~~~
    UNDER CONSTRUCTION
    ~~~

## Protocol

We are currently working on developing our Generic Acquisition Protocol for Whole-Body Quantitative MRI of Muscle. You can access the Google doc [here](https://docs.google.com/document/d/1q7AAnPEr7Rj5gb9d_mLrRnAiav1f32J-RPswvOPk5xE/edit?usp=sharing). To collaborate on the generic acquisition protocol, please contact [Kenneth Weber](mailto:kenweber@stanford.edu), [Eddo Wesselink](mailto:eddo_wesselink@msn.com), [James Elliott](mailto:james.elliott@sydney.edu.au), or [Marnee McKay](mailto:marnee.mckay@sydney.edu.au).

## Data Curation
We strongly recommend following the [Brain Imaging Data Structure (BIDS)](https://bids.neuroimaging.io/) specification for organizing your dataset. Our protools 


### Convert DICOM to BIDS

1. Convert images from [DICOM](https://www.dicomstandard.org/) format to [NIfTI](https://nifti.nimh.nih.gov/) format
    * We recommend using [dcm2niix](https://github.com/rordenlab/dcm2niix) and working with compressed [NIfTI](https://nifti.nimh.nih.gov/) files (nii.gz)
    * Keep the [json](https://en.wikipedia.org/wiki/JSON) sidecar file, which contains imaging parameters

2. Rename the [NIfTI](https://nifti.nimh.nih.gov/) and [json](https://en.wikipedia.org/wiki/JSON) files and organize your dataset to follow the [BIDS](https://bids.neuroimaging.io/) specification 

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

    * sourcedata = contains raw images
    * derivatives = contains segmenation images and other derivatives