Fully automated cardiac shape modeling in tetralogy of Fallot
==============================

Author: Brendan Crabb

Email: brendan.crabb@hsc.utah.edu

Fully-automated, deep-learning based pipeline to generate biventricular cardiac models from raw cine SSFP MRI images. If you use this repository in your research, please cite the following publication:

Govil, S., Crabb, B. T., Deng, Y., Dal Toso, L., Puyol-Antón, E., Pushparajah, K., ... & McCulloch, A. D. (2023). A deep learning approach for fully automated cardiac shape modeling in tetralogy of Fallot. Journal of Cardiovascular Magnetic Resonance, 25(1), 15.


Project Organization
------------

    ├── data
    │   ├── final                <- NIFTI images for use with nnUNet
    │   ├── segmentations        <- nnUNet segmentations for each patient
    │   ├── processed            <- Sorted DICOMS and output files for each patient.
    │   └── raw                  <- The original, immutable data (dicom files).
    |
    ├── models                   <- Trained and serialized models, organized by task
    │   ├──  Landmarks 
    │   ├──  ViewSelection 
    │   ├──  PhaseSelection 
    │   ├──  SliceSelection 
    │   └──  Segmentation
    │
    ├── notebooks                 <- Jupyter notebooks for running end-to-end pipeline
    │
    ├── nnUNet                    <- Local installation of nnUNet (altered to support windows)
    │
    ├── CIM                       <- CIM BiV Modelling v2 code for model fitting
    │
    ├── src                       <- Source code for each step / module
    │   ├── viewselection.py
    │   ├── phaseselection.py  
    │   ├── landmarklocalization.py
    |   ├── annotations.py
    |   ├── utils.py
    │   └── guidepointprocessing.py     
    │
    ├── reports                   <- Generated analysis, saved .csv with view predictions for each series
    │
    └── environment.yml           <- The conda requirements file for reproducing the analysis environment

## Getting Started

### 1.0 Conda Environment

To ensure a working python environment, I recommend creating a new conda environment from the provided environment.yml file. Conda environment files are provided for both windows and ubuntu, please select the appropriate file for your OS. To create the environment, enter the following commands into the terminal: 

```
> conda env create -f environment_windows.yml
```
    
The conda environment will be named "CAP". If you are using an Ubuntu OS, use the environment_ubuntu.yml file. Activate it by running:

```
> conda activate CAP
```

Next, it is necessary to install the local version of nnUNet. To do so, I suggest using pip and the provided setup.py files by entering the code below:

```
> cd nnunet
> pip install -e .
```

### 2.0 Download Trained Models

Download the trained models for each step from the following linK:

https://uofu.box.com/s/wlhvxcyx09wliz9oadpt763ggq9h0rua

In the models folder, each step in the end-to-end pipeline has its own subfolder with the trained model. New models can be added to these folders and called from the corresponding Jupyter notebook if desired. 

### 3.0 End-to-end Pipeline

Once you have a working python environment, and the trained models have been downloaded, you can run the end-to-end pipeline using the provided jupyter notebook, 1.1-BTC-FullAutoCAP.ipynb. This notebook performs view classification, phase selection, SAX slice selection, landmark localization, segmentation, and guidepoint extraction. 

### 4.0 Mesh Fitting

The guidepoint files from the previous step will be copied over into the CIM/BiV_Modelling_v2/test_data/ folder. The provided code performs patient-specific biventricular mesh customization. 

Documentation: https://github.kcl.ac.uk/pages/YoungLab/BiV_Modelling/

How to use:

- before running: set parameters in config_parameters.py
- run Extraction/process_gpFiles to clean LAX contours
- perform_fit.py: performs the biventricular fitting. Prior to use, make sure to set the "main_path" variable within the script. 

For more details, please see the provide readme files in the CIM subfolder. 

