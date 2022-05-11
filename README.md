Towards fully automated cardiac statistical modeling
==============================

Fully-automated, deep-learning based pipeline to generate biventricular cardiac models from raw cine SSFP MRI images.

Author: Brendan Crabb

Email: brendan.crabb@hsc.utah.edu


Project Organization
------------

    ├── data
    │   ├── processed      <- Data sorted according to MRI view.
    │   └── raw            <- The original, immutable data (dicom files).
    |
    ├── models             <- Trained and serialized models, organized by task
    │   ├──  Landmarks 
    │   ├──  ViewSelection 
    │   ├──  PhaseSelection 
    │   └── SliceSelection 
    │
    ├── notebooks          <- Jupyter notebooks for
    │
    ├── reports            <- Generated analysis, saved .csv with view predictions for each series
    │
    └── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
                              generated with `pip freeze > requirements.txt`

## Getting Started

### 1.0 Conda Environment

To ensure a working python environment, it is recommended to create a new conda environment from the provided requirements.txt file. To do so, enter the following commands into the terminal: 

```
conda env create -f environment.yml
```
    
The conda environment will be named "CAP". Activate it by running:

```
conda activate CAP
```

### 2.0 Download Trained Models

Download the trained models for each step from the following linK:

https://uofu.box.com/s/wlhvxcyx09wliz9oadpt763ggq9h0rua

In the models folder, each step in the end-to-end pipeline has its own subfolder with the trained model. New models can be added to these folders and called from the corresponding Jupyter notebook if desired. 

### 3.0 End-to-end Pipeline

Once you have a working python environment, and the trained models have been downloaded, you can run the end-to-end pipeline using the provided jupyter notebook, 1.1-BTC-FullAutoCAP.ipynb.
