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
