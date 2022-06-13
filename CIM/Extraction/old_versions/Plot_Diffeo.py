from BiVFitting import *
import numpy as np
from plotly.offline import plot
import plotly.graph_objs as go
import pandas as pd
import os

#list_to_plot = pd.read_csv('/media/rb20/RB10Tb/CMR_43K_project/Model_qc/Cases_to_mReview_90.csv')
list_to_plot = ['1015864','2751236','3030556','5432815','6002358','1681665']
#list_to_plot = os.listdir('/home/rb20/Desktop/Lauras_UKB_cases/')
frame_df = pd.read_csv('/media/rb20/RB10Tb/CMR_43K_project/Case_ID_and_frame_44k.csv')

#for index, row in list_to_plot.iterrows():
for x in list_to_plot: #['eid']:
    if x[-6:-4] == 'ED':
        pass
    else:
        case_ID = x[:7]  # str(int(row['Case']))
        try:
        #print(x)
            timeframe_row = frame_df.loc[frame_df['feid'] == int(case_ID)] #1-50 from frame dict
            timeframe = timeframe_row['ED_frame'].item() #row['ED_frame']
        except:
            timeframe = 50
        if timeframe == 'None':
            timeframe = 50
        if os.path.exists('/home/rb20/Desktop/problem_cases_for_laura/' + case_ID + '_RB_ED' + '_plot.html'):
            pass
        else:
        #case_ID = str(x)
            print(case_ID)
            plotname = '/home/rb20/Desktop/problem_cases_for_laura/' + case_ID + '_RB_ED' + '_plot.html'
            case = '/media/rb20/RB10Tb/Biobank_Models_EDES_43K/' + case_ID + '_Model_file_diffeo_ED.txt'
            #fp = '/home/rb20/Desktop/ForUKBB_contour_comparison/GPFiles/' + case_ID + '/' + 'GPFile_ED.txt'
            fp = '/media/rb20/RB10Tb/Contours_44K/GPFiles/' + case_ID + '/'
            filename = fp + 'GPFile_proc.txt'  # only proc for old
            filenameInfo = fp + 'SliceInfoFile_proc.txt'

            model_path = '/home/rb20/Desktop/BiVFitting-master(5)/BiVFitting-master/model'
            #filename = fp #+ '/GPFile_proc.txt' #only proc for old
            #filenameInfo = '/home/rb20/Desktop/CharleneModelsRVLVPaper/Models/New_patient_position' + case_ID + '.txt'#fp + '/SliceInfoFile_proc.txt'

            shifting_model = BiventricularModel(model_path,case)
            fitted_nodes = np.loadtxt(case).astype(float)
            shifting_model.update_control_mesh(fitted_nodes)

            contours_to_plot = [ContourType.LAX_RA,
                                ContourType.SAX_RV_FREEWALL, ContourType.LAX_RV_FREEWALL,
                                ContourType.SAX_RV_SEPTUM, ContourType.LAX_RV_SEPTUM,
                                ContourType.SAX_LV_ENDOCARDIAL,
                                ContourType.SAX_LV_EPICARDIAL, ContourType.RV_INSERT,
                                ContourType.APEX_POINT, ContourType.MITRAL_VALVE,
                                ContourType.TRICUSPID_VALVE,
                                ContourType.SAX_RV_EPICARDIAL, ContourType.LAX_RV_EPICARDIAL,
                                ContourType.LAX_LV_ENDOCARDIAL, ContourType.LAX_LV_EPICARDIAL,
                                ContourType.LAX_RV_EPICARDIAL, ContourType.SAX_RV_OUTLET,
                                ContourType.PULMONARY_PHANTOM, ContourType.AORTA_VALVE,
                                ContourType.PULMONARY_VALVE, ContourType.TRICUSPID_PHANTOM,
                                ContourType.AORTA_PHANTOM, ContourType.MITRAL_PHANTOM
                                ]

            data_set = GPDataSet(filename,filenameInfo, case, 5, int(timeframe)) #18 is ES (RV) (timeframe 0-49)
            data_set.sinclaire_slice_shifting()
            contourPlots = data_set.PlotDataSet(contours_to_plot)
            model = shifting_model.PlotSurface("rgb(0,127,0)", "rgb(0,0,127)", "rgb(127,0,0)","Initial model", "all")
            data = contourPlots + model
            plot(go.Figure(data),filename=plotname,auto_open=False)
        #except:
        #    pass
        #plotname = case_ID + '_ml_' + str(timeframe) + '_plot.html'
        #case = '/home/rb20/Desktop/CharleneModelsRVLVPaper/Charlene_models/Models_ED/' \
        #    + case_ID + '_Model_file_diffeo_ED.txt'
        #fp = '/media/rb20/RB10Tb/Contours_27K/GPFiles/' + case_ID + '/'
        #filename = fp + 'GPFile_proc.txt'  # only proc for old
        #filenameInfo = fp + 'SliceInfoFile_proc.txt'
        #shifting_model = BiventricularModel(model_path, case)
        #fitted_nodes = np.loadtxt(case).astype(float)
        #shifting_model.update_control_mesh(fitted_nodes)

        #data_set = GPDataSet(filename, filenameInfo, case, 5, timeframe)  # 18 is ES (RV) (timeframe 0-49)
        #data_set.sinclaire_slice_shifting()
        #contourPlots = data_set.PlotDataSet(contours_to_plot)
        #model = shifting_model.PlotSurface("rgb(0,127,0)", "rgb(0,0,127)", "rgb(127,0,0)", "Initial model", "all")
        #data = contourPlots + model
        #plot(go.Figure(data), filename=plotname, auto_open=False)