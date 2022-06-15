
#Author: Laura Dal Toso 
#Date: 31 May 2022
#Based on work by: Anna Mira, Richard Burns
#------------------------------------------------------------
#Use this script to post process GPFiles and SliceInfo files, and 
#check that all required labels are present
#Before running: 
#- check relative paths
#- check that variables 'contour_file' and 'metadata_file' match your files
#- comment functions that are not needed. 
#    (i.e. if septum points are present, comment 'find_timeframe_septum')

#------------------------------------------------------------


import Contours as cont
from CVI42XML import *
from pathlib import Path
import time

def CleanGPFile(folder, **kwargs):

    '''
    This function perfoms post- processing operations on GPFiles and SliceInfoFiles
    Input: path to folder containing the SliceInfoFile and GPFile
    Otput: new GPFiles and SLiceInfoFiles that can be used as input for the BiVFitting module

    '''

    if 'iter_num' in kwargs:
        # if many processes are run in parallel, assign each one to a different CPU
        iter_num = kwargs.get('iter_num', None)
        pid = os.getpid()
        # assign a new process ID and a new CPU to the child process 
        os.system("taskset -cp %d %d" %(iter_num, pid))
    
    #extract patient name
    case =  os.path.basename(os.path.normpath(folder))
    print('case: ', case )
    frames = ['ES', 'ED']

    for frame in frames:
        contour_file = os.path.join(folder, 'GP_'+frame+'.txt') 
        metadata_file = os.path.join(folder,'SliceInfo.txt')
        
        time_frames = [1]   #ED and ES frames for this example case, taken from SciReport

        #total_stack = ImageStack(input_dir, dicom_extension='dcm')

        contours  = cont.Contours()

        contours.read_gp_files(contour_file,metadata_file,time_frame=time_frames)
        contours.clean_LAX_contour(time_frame=time_frames)

        cvi_cont = CVI42XML()
        cvi_cont.contour = contours

        output_gpfile = os.path.join('../BiV_Modelling_v2/test_data/'+os.path.basename(folder),'GPFile'+frame+'_ldt.txt')
        output_metafile = os.path.join('../BiV_Modelling_v2/test_data/'+os.path.basename(folder),'SliceInfoFile_ldt.txt')


        cvi_cont.export_contour_points(output_gpfile)
        cvi_cont.export_dicom_metadata(output_metafile)


def split_in_chunks(a, n):
    '''
    This function splits the list a in n chunks.
    Otput: list made of n lists

    '''
    k, m = divmod(len(a), n)
    return list(a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
  

def split_and_run(cases_list, workers):
    '''
    This function splits the total workload in chunks of equal size and runs the code in parallel.
    Input:
    cases_folder = list of paths to GPFiles
    workers = number of CPUs to be used
    '''

    # split the total workload in chunks of equal size
    n_chunks = int(np.ceil(len(cases_list)/workers))

    print('TOT CASES:', len(cases_list))
    print('TOT CPUs selected: ', workers )
    print('-----> DATA WILL BE SPLIT INTO ', n_chunks, ' chunks')
    
    # create a Log file where to store failed cases
    #FailedCases = Path(os.path.join('./results/' ,'FailedCases.txt'))
    #FailedCases.touch(exist_ok=True)
    if workers == 1:
        print('1 worker')
        [CleanGPFile(folder) for folder in cases_list]

    if n_chunks <=1:

        # spawn a number of child processes equal to the number of patients in each chunk
        with concurrent.futures.ProcessPoolExecutor(max_workers= workers) as executor:
            #the CPU affinity is changed by perform_fitting, so that each child process is assigned to one CPU 
     
            for i,folder in enumerate(cases_list):   
                results = executor.submit(CleanGPFile, folder, iter_num = i) 

    if n_chunks >1 and workers>1:
        # split data in n chunks:
        split_folders = split_in_chunks(cases_list, n_chunks)
        
        # create a Log file where to store failed cases
        FailedCases = Path(os.path.join('../results/' ,'FailedCases.txt'))
        FailedCases.touch(exist_ok=True)   

        for subfolders in split_folders:
            with open(FailedCases, 'a') as f:
                f.write('In this chunk, cases: '+ str(subfolders) +'\n') 
            
            futures = []
            # spawn a number of child processes equal to the number of patients in each chunk
            with concurrent.futures.ProcessPoolExecutor(max_workers= workers) as executor:
 
                for i,folder in enumerate(subfolders):
                    results = executor.submit(CleanGPFile, folder, iter_num = i)
                    #print( results.result()) #do this is you want to interrupt program when case fails
                    #futures.append( (results, os.path.basename(os.path.normpath(folder))))



if __name__ == '__main__':

    
    startLDT = time.time()
    workers = 1 # number of CPUs to be used in parallel to process the data

    # Create some data
    working_dir = '../BiV_Modelling_v2/'   #edit to your relative path to this data

    cases_folder = os.path.join(working_dir, 'test_data')
    cases_list = [os.path.join(cases_folder, batch) for batch in os.listdir(cases_folder)]

    # this lists all the subfolders in the test_data folder
    results = split_and_run(cases_list, workers)
    print('TOTAL TIME: ', time.time()-startLDT)
