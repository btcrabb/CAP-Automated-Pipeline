Extraction Module
-----------------------------------------------
Author: Laura Dal Toso 
Date: 31 May 2022
Based on work by: Anna Mira, Richard Burns

-----------------------------------------------

Required subfolders:

- CVI42Extraction
- mesh_tools

-----------------------------------------------
Files in main folder: 
- extract_gp_points_kcl_RB : builds SliceInfoFile and GPFile
- process_gpFiles : post-process the GPFiles

-----------------------------------------------
How to use:
- extract_gp_points_kcl_RB: use this module to build GPFile and SliceInfoFile. Input is a .cvi42 file containing contours, and a txt file containing the dicom metadata. 

- process_gpfiles: use this as a second step, to post-process the newly created SliceInfoFile and GPFile. Tune parameter 'workers' to set the number of CPUs to be used in parallel. The function CleanGPFile: deletes the guide points located between two valve points, and it can extract septum, rv inserts, apex and valve points. **Note: Comment some of these functions if the labels are already present in GPFile**







