# Input: 3D contours
# Output: Fitted model

#!/usr/bin/env python3
import os
from plotly.offline import  plot
import plotly.graph_objs as go
import numpy as np
from BiVFitting import BiventricularModel
from BiVFitting import GPDataSet
from BiVFitting import ContourType
from BiVFitting import MultiThreadSmoothingED, SolveProblemCVXOPT
from BiVFitting import plot_timeseries
import time
import pandas as pd 
from pathlib import Path

from config_params import * 

#This list of contours_to _plot was taken from Liandong Lee
contours_to_plot = [ContourType.LAX_RA, ContourType.LAX_RV_ENDOCARDIAL,
                    ContourType.SAX_RV_FREEWALL, ContourType.LAX_RV_FREEWALL,
                    ContourType.SAX_RV_SEPTUM, ContourType.LAX_RV_SEPTUM,
                    ContourType.SAX_LV_ENDOCARDIAL,
                    ContourType.SAX_LV_EPICARDIAL, ContourType.RV_INSERT,
                    ContourType.APEX_POINT, ContourType.MITRAL_VALVE, ContourType.PULMONARY_VALVE,
                    ContourType.TRICUSPID_VALVE, ContourType.AORTA_VALVE,
                    ContourType.SAX_RV_EPICARDIAL, ContourType.LAX_RV_EPICARDIAL,
                    ContourType.LAX_LV_ENDOCARDIAL, ContourType.LAX_LV_EPICARDIAL,
                    ContourType.LAX_RV_EPICARDIAL, ContourType.SAX_RV_OUTLET,
                    ContourType.AORTA_PHANTOM, ContourType.TRICUSPID_PHANTOM,
                    ContourType.MITRAL_PHANTOM, ContourType.PULMONARY_PHANTOM,
                    ContourType.LAX_EPICARDIAL, ContourType.SAX_EPICARDIAL
                    ]

def perform_fitting(folder, **kwargs):
    #performs all the BiVentricular fitting operations
    frames = ['ED', 'ES']
    for frame in frames:
        #try:
                
            if 'id_Frame' in kwargs:
                # acquire .csv file containing patient_id, ES frame number, ED frame number if present
                case_frame_dict = kwargs.get('id_Frame', None)

            # define the path to GPFile and to SliceInfoFile
            # THIS SHOULD BE CHANGED if files are named differently
            filename = os.path.join(folder, 'GPFile'+frame+'_ldt.txt') 
            filenameInfo = os.path.join(folder,'SliceInfoFile_ldt.txt')

            print(filename)

            # extract the patient name from the folder name
            case =  os.path.basename(os.path.normpath(folder))
            print('case: ', case )
            output_folder = './results/' + case
            try:
                os.makedirs(output_folder , exist_ok= True)
            except: raise ValueError

            # create a Log File where to store fitting errors
            Errorfile = Path(os.path.join(output_folder ,'ErrorFile.txt'))
            Errorfile.touch(exist_ok=True)
            Shiftfile = Path(os.path.join(output_folder ,'Shiftfile.txt'))
            Shiftfile.touch(exist_ok=True)
    
            with open(Errorfile, 'w') as f:
                f.write('Log for patient: '+ case+'\n')    

            #read all the frames from the GPFile 
            all_frames = pd.read_csv(filename, sep = '\t')
            frames_to_fit = np.unique([i[6] for i in all_frames.values])   
            print(frames_to_fit)             

            # The next lines are used to measure shift using only a key frame
            if measure_shift_EDonly == True:

                print('shift measured only at ED frame')

                ED_dataset = GPDataSet(os.path.join(folder, 'GPFileED_ldt.txt') ,filenameInfo, case, sampling = sampling, time_frame_number = 1)
                result_ED = ED_dataset.sinclaire_slice_shifting( frame_num = 1) 
                shift_ED = result_ED[0]
                pos_ED = result_ED[1]
                #np.save(os.path.join(output_folder, 'shift.txt'), shift_ED)
                with  open(Shiftfile, "w") as file:
                        file.write('shift measured only at ED: frame '+ str(1)+'\n')
                        file.write(str(shift_ED))
                        file.close()        

    
            #initialise time series lists
            TimeSeries_step1 = []
            TimeSeries_step2 = []
        
            print('FITTING OF  ', str(case), '----> started \n')

            for idx,num in enumerate(sorted(frames_to_fit)):
                    num = int(num) #frame number
                    print('frame num', num)

                    Modelfile = Path(os.path.join(output_folder , str(case)+'_Model_Frame_'+frame+'.txt'))
                    Modelfile.touch(exist_ok=True)  

                    with open(Errorfile, 'a') as f: 
                        f.write('\nFRAME #' +str(int(num))+'\n')

                    model_path = './model'
                    data_set = GPDataSet(filename,filenameInfo, case, sampling = sampling, time_frame_number = num)
                    biventricular_model = BiventricularModel(model_path, case)                        
                    model = biventricular_model.plot_surface("rgb(0,127,0)",  
                                                            "rgb(0,0,127)",
                                                            "rgb(127,0,0)",
                                                            surface = "all")  


                    if measure_shift_EDonly == True:
                        # apply shift measured previously using ED frame
                        data_set.apply_slice_shift(shift_ED, pos_ED)

                    else: 
                        # measure and apply shift to current frame
                        shiftedSlice = data_set.sinclaire_slice_shifting(Errorfile, int(num)) 
                        shiftmeasure = shiftedSlice[0]

                        if idx == 0:  
                            with  open(Shiftfile, "w") as file:
                                file.write('Frame number:  ' + str(num)+'\n')
                                file.write(str(shiftmeasure))
                                file.close()     

                        else:  
                            with  open(Shiftfile, "a") as file:
                                file.write('\nFrame number:  ' + str(num)+'\n')
                                file.write(str(shiftmeasure))
                                file.close()     
                        pass 
                    
                    #model_path = "./model"

                    biventricular_model.update_pose_and_scale(data_set)

                    # # perform a stiff fit
                    # displacement, err = biventricular_model.lls_fit_model(weight_GP,data_set,1e10)
                    # biventricular_model.control_mesh = np.add(biventricular_model.control_mesh,
                    #                                           displacement)
                    # biventricular_model.et_pos = np.linalg.multi_dot([biventricular_model.matrix,
                    #                                                   biventricular_model.control_mesh])
                    # displacements = data_set.SAXSliceShiffting(biventricular_model)

                    contourPlots = data_set.PlotDataSet(contours_to_plot)    


                    #plot(go.Figure(contourPlots))
                    data = contourPlots

                    #plot(go.Figure(data),filename=os.path.join(folder, 'pose_fitted_model_Frame'+str(int(num))+'.html'), auto_open=False) 

                    # Generates RV epicardial point if they have not been contoured
                    # (can be commented if available) used in LL 
                    #rv_epi_points,rv_epi_contour, rv_epi_slice = data_set.create_rv_epicardium(
                        #rv_thickness=3)


                    # Generates 30 BP_point phantom points and 30 tricuspid phantom points.
                    # We do not have any pulmonary points or aortic points in our dataset but if you do,
                    # I recommend you to do the same.

                    mitral_points = data_set.create_valve_phantom_points(30, ContourType.MITRAL_VALVE)
                    tri_points = data_set.create_valve_phantom_points(30, ContourType.TRICUSPID_VALVE)
                    print('Pulmonary phantom: \n')
                    pulmonary_points = data_set.create_valve_phantom_points(20, ContourType.PULMONARY_VALVE)
                    aorta_points = data_set.create_valve_phantom_points(20, ContourType.AORTA_VALVE)
                

                    # Example on how to set different weights for different points group (R.B.)
                    data_set.weights[data_set.contour_type == ContourType.MITRAL_PHANTOM] = 2
                    data_set.weights[data_set.contour_type == ContourType.AORTA_PHANTOM] = 1
                    data_set.weights[data_set.contour_type == ContourType.PULMONARY_PHANTOM] = 1
                    data_set.weights[data_set.contour_type == ContourType.TRICUSPID_PHANTOM] = 1

                    data_set.weights[data_set.contour_type == ContourType.APEX_POINT] = 1
                    data_set.weights[data_set.contour_type == ContourType.RV_INSERT] = 1
                    
                    data_set.weights[data_set.contour_type == ContourType.MITRAL_VALVE] = 1
                    data_set.weights[data_set.contour_type == ContourType.AORTA_VALVE]= 1
                    data_set.weights[data_set.contour_type == ContourType.PULMONARY_VALVE] = 1                
                    
                    # Perform linear fit
                    MultiThreadSmoothingED(biventricular_model,weight_GP, data_set, Errorfile)
    
                    # Plot results
                    model = biventricular_model.plot_surface("rgb(0,127,0)", "rgb(0,0,127)","rgb(127,0,0)","all")   
                    data = model + contourPlots
                    #TimeSeries_step1.append([data, num])

                    # Perform diffeomorphic fit
                    SolveProblemCVXOPT(biventricular_model,data_set,weight_GP,low_smoothing_weight,
                                                        transmural_weight,Errorfile)

                    # Plot final results
                    model = biventricular_model.plot_surface("rgb(0,127,0)", "rgb(0,0,127)", "rgb(127,0,0)","all")
                    data = model + contourPlots
                    TimeSeries_step2.append([data, num])
                    
                    #plot(go.Figure(data),filename=os.path.join(folder, 'step2_fitted_model_Frame'+str(int(num))+'.html'), auto_open=False)
                    
                    # save results in .txt format, one file for each frame
                    ModelData = {'x': biventricular_model.control_mesh[:,0], 'y': biventricular_model.control_mesh[
                        :,1], 'z': biventricular_model.control_mesh[:,2], 'Frame': [num] * len(biventricular_model.control_mesh[:,2])}
                    
                    Model_Dataframe = pd.DataFrame(data=ModelData)
                    with  open(Modelfile, "w") as file:
                            file.write(Model_Dataframe.to_string(header=True, index=False))

            # if you want to plot time series in html files uncomment the next line(s)
            plot_timeseries(TimeSeries_step2, output_folder, 'TimeSeries_'+frame+'.html')
            DoneFile = Path(os.path.join(output_folder ,'Done.txt'))
            DoneFile.touch(exist_ok=True)   

        #except KeyboardInterrupt:
        #    raise KeyboardInterruptError()




if __name__ == '__main__':

    
    startLDT = time.time()
    #pid = os.getpid()
    #os.system("taskset -cp %d %d" %(66, pid))

    main_path = 'E:/CAP/CAP-FullAutomation/CIM/BiV_Modelling_v2/'          ### folder in use

    cases_folder = os.path.join(main_path, 'test_data')
    cases_list = [os.path.join(cases_folder, batch) for batch in os.listdir(cases_folder)]


    results = [perform_fitting (folder) for folder in cases_list]

    print('TOT CASES:', len(cases_list))
    
    print('TOTAL TIME: ', time.time()-startLDT)

