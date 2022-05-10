Towards fully automated cardiac statistical modeling
==============================
Author: Brendan Crabb
Email: brendan.crabb@hsc.utah.edu

Fully-automated, deep-learning based pipeline to generate biventricular cardiac models from raw cine SSFP MRI images.

Project Organization
------------

    ├── data
    │   ├── processed      <- Data sorted according to MRI view.
    │   ├── raw            <- The original, immutable data (dicom files).
    |
    ├── models             <- Trained and serialized models (VGG-19, ResNet50, and Xception)
    │
    ├── notebooks          <- Jupyter notebooks for
    │
    ├── reports            <- Generated analysis, saved .csv with view predictions for each series
    │
    └── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
                              generated with `pip freeze > requirements.txt`
