
#Author: Laura Dal Toso 
#Date: 31 May 2022

#------------------------------------------------------------
#Use this script to convert the old GP files and SliceInfo files to new formats, 
# which are compatible with the new version of the BiVFitting module. 

#Before running: 
#- check relative paths
#- check that GP and SliceInfo filename match your input data
#

#------------------------------------------------------------

import os
from pathlib import Path

from BiVFitting import BiventricularModel
from BiVFitting import GPDataSet
from BiVFitting import ContourType
from BiVFitting import MultiThreadSmoothingED, SolveProblemCVXOPT
from BiVFitting import plot_timeseries

main_path = '.'          ### folder in use

cases_folder = os.path.join(main_path, 'test_data') # folder where test data are located
cases_list = [os.path.join(cases_folder, batch) for batch in os.listdir(cases_folder)] # list all elements

    
for folder in cases_list:  
    file = os.path.join(folder, 'SliceInfo.txt')   #define input SliceInfo file

    with open(file , 'r') as f:
        # read SliceInfo file
        Lines = f.readlines()

        old_lines = []
        for i, line in enumerate(Lines):
            old_line = line.strip()
            old_lines.append(old_line)


        # split the list in groups of eight
        N = 8
        subList = [old_lines[n:n+N] for n in range(0, len(old_lines), N)]
        
        #define output Slice Info file
        SliceInfofile = Path(os.path.join(folder ,'SliceInfoFile.txt'))
        if os.path.exists(SliceInfofile):
            os.remove(SliceInfofile)
        SliceInfofile.touch()        
        
        # create dictionary containing all needed metadata
        Slices_View_Dict = {}
        for i in subList:

            SliceInfoDict = {}
            row1_elements = [x.strip() for x in i[0].split(' ')]
            SliceInfoDict[row1_elements[0]] = row1_elements[1] #Instance UID
            SliceInfoDict['View'] = row1_elements[2] #LA or SA

            row2_elements = [x.strip() for x in i[1].split(' ')]
            SliceInfoDict[row2_elements[0]] = row2_elements[2] #SLiceID
            
            Slices_View_Dict[row2_elements[2]] = row1_elements[2]
            SliceInfoDict[i[2]] = i[3].replace(" ", "\t")
            SliceInfoDict[i[4]] = i[5].replace(" ", "\t")
            SliceInfoDict[i[6]] = i[7].replace(" ", "\t")
    
            with open(SliceInfofile , 'a') as f:
                f.write(SliceInfoDict[row1_elements[0]]+'\t'+'frameID:\t'+ SliceInfoDict[
                    row2_elements[0]]+'\ttimeFrame\t1\tImagePositionPatient\t'+ SliceInfoDict[
                        i[2]]+'\tImageOrientationPatient\t'+SliceInfoDict[
                            i[4]]+'\tPixelSpacing\t'+SliceInfoDict[i[6]]+'\n')

        
        print('SliceInfoFiles completed')

    # define frames for which the guide point files are available
    frames = ['ED','ES']

    for frame in frames: 
        #acquire old guide point file
        old_GP = os.path.join(folder, 'GP_'+frame+'.txt') 
        
        new_lines_GP  = []
        with open(old_GP, 'r') as f:
            Lines = f.readlines()

            # reformat file, changing labels and adding weight and time_frame columns
            for i, line in enumerate(Lines):
                    old_line = line.strip()
                    SliceNumber = int(old_line[-2:])
                    View = Slices_View_Dict[str(SliceNumber)]
                    old_line = old_line.replace(" ", "\t")
                    old_line = old_line.replace("saendocardialContour", View+"X_LV_ENDOCARDIAL")
                    old_line = old_line.replace("RVFW", View+"X_RV_FREEWALL")
                    old_line = old_line.replace("saepicardialContour", View+"X_EPICARDIAL")
                    old_line = old_line.replace("RVS", View+"X_RV_SEPTUM")
                    old_line = old_line.replace("BP_point", "MITRAL_VALVE")
                    old_line = old_line.replace("Tricuspid_Valve", "TRICUSPID_VALVE")
                    old_line = old_line.replace("Aorta", "AORTA_VALVE")
                    old_line = old_line.replace("Pulmonary", "PULMONARY_VALVE")
                    old_line = old_line.replace("LA_Apex_Point", "APEX_POINT")
                    old_line = old_line.replace("RV_insert", "RV_INSERT")
                    new_lines_GP.append(old_line + '\t 1.0 \t 1.0')
        
        #create new guide point file
        new_GP = Path(os.path.join(folder ,'GP_'+frame+'_ldt.txt'))
        new_GP.touch(exist_ok=True)

        with open(new_GP, 'w') as f:
            f.write('x \ty \tz \tcontour type \tframeID \tweight \ttime frame\n')
            for item in new_lines_GP:
                f.write("%s\n" % item)

