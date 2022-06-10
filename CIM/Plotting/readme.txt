
Author: LDT
Date: 26 May 2022


Folders:
- BiVFitting: contains modules required for biventricular fitting
- model: contains the model files required by the BiVFitting routines

Files: 
- Plot_html: to plot html files from the .txt files containing mesh coordinates
- Split_Model_Files: Use to split a single .txt file containing mesh coordinates for all frames to many .txt files, one for each time frame.
    This step is necessary if the goal is to visualise time series using Paraview. 
    Paraview will automatically acquire separate .txt files as a time series if consecutive frame numbers are included in the file name.

-vtk: Run this after Split_Model_Files. 
    This script saves the meshes and guide points in separate .vtk files, that need to be loaded separately in Paraview.