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

#### Download Trained Models

Download the trained models for each step from the following linK:

https://uofu.box.com/s/wlhvxcyx09wliz9oadpt763ggq9h0rua

In the models folder, each step in the end-to=end pipeline has its own subfolder with the trained model. New models can be added to these folders and called from the corresponding Jupyter notebook if desired. 
