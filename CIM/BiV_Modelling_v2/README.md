# BiV_Modelling_parallel
Parallel CPU computation using the biventricular model 

-----------------------------------------------
Author: Laura Dal Toso 
Date: 4 Feb 20022
Based on work by: Anna Mira, Liandong Lee, Richard Burns

-----------------------------------------------

This code performs patient-specific biventricular mesh customization. 

The process takes place in 2 steps:
1. correction of breath-hold misregistration between short-axis slices, and 
2. deformation of the subdivided template mesh to fit the manual contours while preserving 
the topology of the mesh.

Documentation: https://github.kcl.ac.uk/pages/YoungLab/BiV_Modelling/


-----------------------------------------------
Required subfolders:

- BiVFitting: contains the code that performs patient-specific biventricular mesh customization. 
- model: contains .txt files required by the fitting modules
- results: output folder
- test_data: contains one subfolder for each patient. Each subfolder contains the GPFile and SliceInfoFile realtive to one patient.

-----------------------------------------------
Files in main folder: 

- config_params: parameters needed for the simulation
- perform_fit: script that contains the routine to perform the biventricular fitting


-----------------------------------------------
How to use:

- before running: set parameters in config_parameters.py
- run Extraction/process_gpFiles to clean LAX contours
- perform_fit.py: performs the biventricular fitting

