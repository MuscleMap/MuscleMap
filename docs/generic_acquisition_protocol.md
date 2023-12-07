# MuscleMap: Generic Acquisition Protocol for Whole-Body Quantitative MRI of Muscle

## Table of contents
1. [How to Contribute](#howto)
2. [Dataset](#2dataset)
3. [Analysis pipeline](#3analysis-pipeline)
    * [3.1.Installation](#31installation)
4. [TEST](#participantstsv)

## 1. How to Contribute <a name="howto"></a>
If your site is interested in contributing to the generic acquisition protocol, please coordinate with [Kenneth Weber](mailto:kenweber@stanford.edu).

### 1.2 Get the Generic Acquisition Protocol
The Generic Acquisition Protocol by manufacturer and model is available [here](https://github.com/MuscleMap/MuscleMap/tree/main/protocol). Follow the standard operating procedures available here.

### 1.4 Collect the data
We are easking each site to scan six healthy participants (3 males and 3 females) between 20 and 40 years of age. If a single site has multiple scanners of varying manufacturer or model, a single site can contribute multiple datasets.

#### 1.4.1 Use the Standard Operating Procedures

#### 1.4.2 Ethics and Anonymization
Ethics approval is required by each site, and each participant should provide written consent, which should include the their anonymized data can be made publicly-available in a data repository.

## 2. DICOM to BIDS Conversion
To facilitate curation, sharing, and processing of data, we will use the `BIDS` standard (http://bids.neuroimaging.io/). An example of the data structure for one center is shown below:

    data-Example
    │── sourcedata
    |   ├── dataset_description.json
    |   ├── participants.json
    |   ├── participants.tsv
    |   ├── sub-example01
    |   ├── sub-example02
    |   ├── sub-example03
    |   ├── sub-example04
    |   ├── sub-example05
    |   ├── sub-example06
    │   |── anat
    │       ├── sub-example06_T1w.json
    │       ├── sub-example06_T1w.nii.gz
    │       ├── sub-example06_T2star.json
    │       ├── sub-example06_T2star.nii.gz
    │       ├── sub-example06_T2w.json
    │       ├── sub-example06_T2w.nii.gz
    │       ├── sub-example06_flip-1_mt-off_MTS.json
    │       ├── sub-example06_flip-1_mt-off_MTS.nii.gz
    │       ├── sub-example06_flip-1_mt-on_MTS.json
    │       ├── sub-example06_flip-1_mt-on_MTS.nii.gz
    │       ├── sub-example06_flip-2_mt-off_MTS.json
    │       └── sub-example06_flip-2_mt-off_MTS.nii.gz
    └── derivatives

### 2.1. Installation of DICOM to BIDS Converter 

#### 2.2.1 Create python environement
~~~
conda create --name dicom2bids python==3.8.12
~~~
#### 2.2.2 Activate environement
~~~
conda activate dicom2bids
~~~
#### 2.2.3. Install requirements
~~~
pip install -r requirements.txt
~~~

### 2.2. Running DICOM to BIDS Converter
~~~
dcm2bids.py -d PATH_TO_DICOM -p sub-ID_SITE -c config_spine.txt -o SITE_spineGeneric
~~~

If you encounter any problem while running the script, please open an issue [here](https://github.com/MuscleMap/MuscleMap/issues) and upload the log file. We will offer support.

### 2.3. Create description.json file

Once you have converted all subjects for the study, create the following files and add them to the data structure:

**dataset\_description.json** (Pick the correct values depending on your system and environment)
    {
        "Name": "Spinal Cord MRI Public Database",
        "BIDSVersion": "1.2.0",
        "InstitutionName": "Name of the institution",
        "Manufacturer": "YOUR_VENDOR",
        "ManufacturersModelName": "YOUR_MODEL",
        "ReceiveCoilName": "YOUR_COIL",
        "SoftwareVersion": "YOUR_SOFTWARE",
        "Researcher": "J. Doe, S. Wonder, J. Pass",
        "Notes": "Particular notes you might have. E.g.: We don't have the ZOOMit license, unf-prisma/sub-01 and unf-skyra/sub-03 is the same subject.
    }

Example of possible values:
- **Manufacturer**: "Siemens", "GE", "Philips"
- **ManufacturersModelName**: "Prisma", "Prisma-fit", "Skyra", "750w", "Achieva"
- **ReceiveCoilName**: "64ch+spine", "12ch+4ch neck", "neurovascular"
- **SoftwareVersion**: "VE11C", "DV26.0", "R5.3", ...

### 2.4. Create participants.json file

**participants.json** (This file is generic, you don't need to change anything there. Just create a new file with this content)

    {
        "participant_id": {
            "LongName": "Participant ID",
            "Description": "Unique ID"
        },
        "sex": {
            "LongName": "Participant gender",
            "Description": "M or F"
        },
        "age": {
            "LongName": "Participant age",
            "Description": "yy"
        },
        "date_of_scan": {
            "LongName": "Date of scan",
            "Description": "yyyy-mm-dd"
        }
    }

### 2.4. Create participants.tsv file  <a name="participantstsv"></a>

**participants.tsv** (Tab-separated values)

    participant_id  sex age date_of_scan    institution_id  institution manufacturer    manufacturers_model_name    receive_coil_name   software_versions   researcher
    sub-example01   F   24  2018-12-07  unf Neuroimaging Functional Unit (UNF), CRIUGM, Polytechnique Montreal  Siemens Prisma-fit  HeadNeck_64 syngo_MR_E11    J. Cohen-Adad, A. Foias
    sub-example02   M   29  2018-12-07  unf Neuroimaging Functional Unit (UNF), CRIUGM, Polytechnique Montreal  Siemens Prisma-fit  HeadNeck_64 syngo_MR_E11    J. Cohen-Adad, A. Foias
    sub-example03   M   22  2018-12-07  unf Neuroimaging Functional Unit (UNF), CRIUGM, Polytechnique Montreal  Siemens Prisma-fit  HeadNeck_64 syngo_MR_E11    J. Cohen-Adad, A. Foias
    sub-example04   M   31  2018-12-07  unf Neuroimaging Functional Unit (UNF), CRIUGM, Polytechnique Montreal  Siemens Prisma-fit  HeadNeck_64 syngo_MR_E11    J. Cohen-Adad, A. Foias
    sub-example05   F   23  2019-01-11  unf Neuroimaging Functional Unit (UNF), CRIUGM, Polytechnique Montreal  Siemens Prisma-fit  HeadNeck_64 syngo_MR_E11    J. Cohen-Adad, A. Foias
    sub-example06   F   27  2019-01-11  unf Neuroimaging Functional Unit (UNF), CRIUGM, Polytechnique Montreal  Siemens Prisma-fit  HeadNeck_64 syngo_MR_E11    J. Cohen-Adad, A. Foias

Once you've created the BIDS dataset, remove any temp folders (e.g., ``tmp_dcm2bids/``) and zip the entire folder. It is now ready for sharing! You could send it to [Kenneth Weber](mailto:kenweber@stanford.edu) via any cloud-based method (Gdrive, Dropbox, etc.).
