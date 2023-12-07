Data collection and organization
================================


Imaging Protocol
-----------------
The "Spine Generic" MRI acquisition protocol is available at `this
link <http://www.spinalcordmri.org/protocols>`__. Each site was instructed to scan six healthy subjects
(3 men, 3 women), aged between 20 and 40 y.o. Note: there was some
flexibility in terms of number of participants and age range.

If your site is interested in contributing to the publicly-available database, please
coordinate with Julien Cohen-Adad.


.. _multi-center-data:

Data conversion: DICOM to BIDS
------------------------------

To facilitate the collection, sharing and processing of data, we use the
`BIDS standard <http://bids.neuroimaging.io/>`__. An example of the data
structure for one center is shown below:

::

    data-multi-subject
    │
    ├── dataset_description.json
    ├── participants.json
    ├── participants.tsv
    ├── sub-ubc01
    ├── sub-ubc02
    ├── sub-ubc03
    ├── sub-ubc04
    ├── sub-ubc05
    ├── sub-ubc06
    │   │
    │   ├── anat
    │   │   ├── sub-ubc06_T1w.json
    │   │   ├── sub-ubc06_T1w.nii.gz
    │   │   ├── sub-ubc06_T2star.json
    │   │   ├── sub-ubc06_T2star.nii.gz
    │   │   ├── sub-ubc06_T2w.json
    │   │   ├── sub-ubc06_T2w.nii.gz
    │   │   ├── sub-ubc06_flip-1_mt-off_MTS.json
    │   │   ├── sub-ubc06_flip-1_mt-off_MTS.nii.gz
    │   │   ├── sub-ubc06_flip-1_mt-on_MTS.json
    │   │   ├── sub-ubc06_flip-1_mt-on_MTS.nii.gz
    │   │   ├── sub-ubc06_flip-2_mt-off_MTS.json
    │   │   └── sub-ubc06_flip-2_mt-off_MTS.nii.gz
    │   │
    │   └── dwi
    │       ├── sub-ubc06_dwi.bval
    │       ├── sub-ubc06_dwi.bvec
    │       ├── sub-ubc06_dwi.json
    │       ├── sub-ubc06_dwi.nii.gz
    │       ├── (sub-ubc06_acq-b0_dwi.json)
    │       └── (sub-ubc06_acq-b0_dwi.nii.gz)
    │
    └── derivatives
        │
        └── labels
            └── sub-ubc06
                │
                ├── anat
                │   ├── sub-ubc06_T1w_RPI_r_seg-manual.nii.gz  <---------- manually-corrected spinal cord segmentation
                │   ├── sub-ubc06_T1w_RPI_r_seg-manual.json  <------------ information about origin of segmentation (see below)
                │   ├── sub-ubc06_T1w_RPI_r_labels-manual.nii.gz  <------- manual vertebral labels
                │   ├── sub-ubc06_T1w_RPI_r_labels-manual.json
                │   ├── sub-ubc06_T2w_RPI_r_seg-manual.nii.gz  <---------- manually-corrected spinal cord segmentation
                │   ├── sub-ubc06_T2w_RPI_r_seg-manual.json
                │   ├── sub-ubc06_flip-2_mt-off_MTS_seg-manual.nii.gz  <-------- manually-corrected spinal cord segmentation
                │   ├── sub-ubc06_flip-2_mt-off_MTS_seg-manual.json
                │   ├── sub-ubc06_T2star_rms_gmseg-manual.nii.gz  <------- manually-corrected gray matter segmentation
                │   └── sub-ubc06_T2star_rms_gmseg-manual.json
                │
                └── dwi
                    ├── sub-ubc06_dwi_moco_dwi_mean_seg-manual.nii.gz  <-- manually-corrected spinal cord segmentation
                    └── sub-ubc06_dwi_moco_dwi_mean_seg-manual.json


To convert your DICOM data folder to the compatible BIDS structure, you need to install
`dcm2bids <https://github.com/cbedetti/Dcm2Bids#install>`__. Once
installed, `download this config
file <https://github.com/spine-generic/spine-generic/blob/master/config_spine.txt>`__
(click File>Save to save the file), then convert your Dicom folder using
the following command (replace xx with your center and subject number):

.. code-block:: bash

  dcm2bids -d PATH_TO_DICOM -p sub-ID_SITE -c config_spine.txt -o SITE_spineGeneric

For example:

.. code-block:: bash

  dcm2bids -d /Users/julien/Desktop/DICOM_subj3 -p sub-milan03 -c ~/Desktop/config_spine.txt -o milan_spineGeneric


A log file is generated under ``tmp_dcm2bids/log/``. If you encounter
any problem while running the script, please `open an
issue <https://github.com/spine-generic/spine-generic/issues>`__ and
upload the log file. We will offer support.

Once you have converted all subjects for the study, create the following
files and add them to the data structure:

**dataset\_description.json** (Pick the correct values depending on your
system and environment)

::

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

**participants.json** (This file is generic, you don't need to change
anything there. Just create a new file with this content)

.. code:: json

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

**participants.tsv** (Tab-separated values)

::

    participant_id  sex age date_of_scan    institution_id  institution manufacturer    manufacturers_model_name    receive_coil_name   software_versions   researcher
    sub-unf01   F   24  2018-12-07  unf Neuroimaging Functional Unit (UNF), CRIUGM, Polytechnique Montreal  Siemens Prisma-fit  HeadNeck_64 syngo_MR_E11    J. Cohen-Adad, A. Foias
    sub-unf02   M   29  2018-12-07  unf Neuroimaging Functional Unit (UNF), CRIUGM, Polytechnique Montreal  Siemens Prisma-fit  HeadNeck_64 syngo_MR_E11    J. Cohen-Adad, A. Foias
    sub-unf03   M   22  2018-12-07  unf Neuroimaging Functional Unit (UNF), CRIUGM, Polytechnique Montreal  Siemens Prisma-fit  HeadNeck_64 syngo_MR_E11    J. Cohen-Adad, A. Foias
    sub-unf04   M   31  2018-12-07  unf Neuroimaging Functional Unit (UNF), CRIUGM, Polytechnique Montreal  Siemens Prisma-fit  HeadNeck_64 syngo_MR_E11    J. Cohen-Adad, A. Foias
    sub-unf05   F   23  2019-01-11  unf Neuroimaging Functional Unit (UNF), CRIUGM, Polytechnique Montreal  Siemens Prisma-fit  HeadNeck_64 syngo_MR_E11    J. Cohen-Adad, A. Foias
    sub-unf06   F   27  2019-01-11  unf Neuroimaging Functional Unit (UNF), CRIUGM, Polytechnique Montreal  Siemens Prisma-fit  HeadNeck_64 syngo_MR_E11    J. Cohen-Adad, A. Foias

Once you've created the BIDS dataset, remove any temp folders (e.g.,
``tmp_dcm2bids/``) and zip the entire folder. It is now ready for
sharing! You could send it to Julien Cohen-Adad via any cloud-based
method (Gdrive, Dropbox, etc.).


Checking acquisition parameters
-------------------------------

To ensure the acquisition protocol was properly followed by each site, we implemented a parameter validator that 
verifies if the pulse sequence parameters match the required ones from the generic protocol (within a tolerance range).
Basic parameters are checked, including: repetition time, echo time, flip angle. These parameters are read from the 
json sidecar file (generated by the DICOM to BIDS conversion). Note that BIDS file naming convention is also checked 
by the validator. If a parameter does not match, a warning message is triggered.

This validator is exposed in this command line interface (CLI) function: **sg_params_checker**. This function is run 
during continuous integration (CI), for each dataset, ensuring valid dataset throughout the life cycle of the project. 

The json file containing the recommended acquisition parameters is located under `/spinegeneric/cli/specs.json`.

Example usage and expected output:

.. code-block:: bash

  sg_params_checker -path-in ~/data-single-subject/
  WARNING: Incorrect FlipAngle: sub-douglas_T2w.nii.gz; FA=120 instead of 180
  WARNING: Incorrect RepetitionTime: sub-mgh_T2w.nii.gz; TR=2 instead of 1.5
  WARNING: Incorrect FlipAngle: sub-tokyoSigna1_T2star.nii.gz; FA=20 instead of 30
  WARNING: Incorrect FlipAngle: sub-tokyoSigna2_T2star.nii.gz; FA=20 instead of 30
  WARNING:sub-ucl_T2star.nii.gz Missing Manufacturer in json sidecar; Cannot check parameters.


Ethics and anonymization
------------------------

Each subject consented to be scanned and to have their anonymized data
put in a publicly-available repository. To prove it, an email from each
participant should be sent to the manager of the database (Julien
Cohen-Adad). The email should state the following: "I am the subject who
volunteered and I give you permission to release the scan freely to the
public domain."

Anatomical scans where facial features are visible (T1w) could be
"defaced" before being collected (at the discretion of the subject).

This can be done automatically using R or manually, in case the automatic
defacing fails.


Automatic defacing with R
^^^^^^^^^^^^^^^^^^^^^^^^^

1. Install `R <https://www.r-project.org/>`__, then open R (type "r" in the Terminal) and install the following dependencies:

.. code-block:: R

  install.packages("sessioninfo")
  install.packages("remotes")
  remotes::install_github("muschellij2/oro.nifti")  # answer "Yes" to "install from source?"
  install.packages("fslr")
  install.packages("argparser")
  install.packages("devtools")
  remotes::install_github("muschellij2/extrantsr")  # choose "1" when prompted

2. Download this repository and install Python's dependencies as instructed in :ref:`getting-started`.

3. Run:

.. code-block:: bash

  sg_deface_using_r -i PATH_TO_BIDS_DATASET -o PATH_TO_DEFACED_BIDS_DATASET -f
  sg_deface_using_r -i PATH_TO_BIDS_DATASET -o PATH_TO_DEFACED_BIDS_DATASET

4. To launch the QC report of the defacing across multiple subjects, run:

.. code-block:: bash

  sg_qc_bids_deface


Manual Defacing
^^^^^^^^^^^^^^^

Automatic defacing might fail in some subjects, so this section explains how
to deface manually. This procedure takes less than a minute per subject. Here
we use FSLeyes but you can use any other NIfTI image editor.

Open FSLeyes and load the T1w scan. Go to **Tools > Edit mode**, Select
the pencil with size 100, deface, then save.

Below is an example of a defaced subject:

.. figure:: _static/example_defacing.png
   :alt: example\_defacing
   :align: center
   :scale: 70%

   Example of manual defacing.

Example of datasets
-------------------

**T1w - sub-vuiisAchieva02**

.. raw:: html

    <iframe src="_static/sub-vuiisAchieva02_T1w.html"  width=800 height=500 style="padding:0; border:0; display: block; margin-left: auto; margin-right: auto"></iframe>

**T2w - sub-milan01**

.. raw:: html
    
    <iframe src="_static/sub-milan01_T2w.html"  width=800 height=700 style="padding:0; border:0; display: block; margin-left: auto; margin-right: auto"></iframe>

**T2star - sub-brnoCeitec01**

.. raw:: html

    <iframe src="_static/sub-brnoCeitec01_T2star.html"  width=800 height=300 style="padding:0; border:0; display: block; margin-left: auto; margin-right: auto"></iframe>

**flip-1_mt-on_MTS - sub-barcelona04**

.. raw:: html

    <iframe src="_static/sub-barcelona04_acq-MTon_MTS.html"  width=800 height=400 style="padding:0; border:0; display: block; margin-left: auto; margin-right: auto"></iframe>

**flip-1_mt-off_MTS - sub-barcelona04**

.. raw:: html

    <iframe src="_static/sub-barcelona04_acq-MToff_MTS.html" width=800 height=400 style="padding:0; border:0; display: block; margin-left: auto; margin-right: auto"></iframe>

**flip-2_mt-off_MTS - sub-barcelona04**

.. raw:: html

    <iframe src="_static/sub-barcelona04_acq-T1w_MTS.html"  width=800 height=400 style="padding:0; border:0; display: block; margin-left: auto; margin-right: auto"></iframe>

**DWI - sub-barcelona04**

.. raw:: html

    <iframe src="_static/sub-barcelona04_dwi.html"  width=800 height=400 style="padding:0; border:0; display: block; margin-left: auto; margin-right: auto"></iframe>
