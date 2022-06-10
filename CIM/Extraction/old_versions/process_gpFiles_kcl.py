#from imaging_tools import ImageStack
#from imaging_tools import DicomViewer

from CVI42Extraction.CVI42XML import *
from CVI42Extraction.Contours import *
import sys
################################################################################


# Create some data
working_dir = '/home/rb20/Desktop/Biven_modelling_pipeline_Circle/Example_data/' #edit to your relative path to this data
case = '1000231'
contour_file = working_dir + '1000231_HPC/GPFile.txt'
metadata_file = working_dir + '1000231_HPC/SliceInfoFile.txt'

time_frames = list(range(1,51)) #ED and ES frames for this example case, taken from SciReport

#total_stack = ImageStack(input_dir, dicom_extension='dcm')

contours  = Contours()

contours.read_gp_files(contour_file,metadata_file,time_frame=time_frames)


try:
    contours.find_timeframe_septum(time_frames=time_frames)
except:
    err = 'Computing septum'
    print('\033[1;33;41m  {0}\t{1}\t\t\t{2}'.format(case, 'Fail', err))


try:
    contours.find_timeframe_septum_inserts(time_frame=time_frames)
except:
    err = 'Computing inserts'
    print('\033[1;33;41m  {0}\t{1}\t\t\t{2}'.format(case, 'Fail',err))

try:
    contours.find_apex_landmark(time_frame=time_frames)
except:
    err = 'Computing apex'
    print('\033[1;33;41m  {0}\t{1}\t\t\t{2}'.format(case, 'Fail',err))

try:
    #contours.find_timeframe_valve_landmarks()
    phases = time_frames
    if 'LAX_LV_EXTENT' in contours.points.keys():
        for index,point in enumerate(contours.get_timeframe_points(
                            'LAX_LV_EXTENT', phases)[1]):
            # the extents has 3 points, for each extent we need to
            # select the first 2 corresponding to the valve
            # the output from get_timeframe_points is already sorted by timeframe
            # therefor we pick the firs to points by timeframe

            # In this dataset the LAX_EXTENT in 3CH is not corresponding to
            # the mitral valve so we need to exclude them
            # if there are aorta points on the same timeframe
            # then is a 3ch and we need to exclude them
            aorta_points,_ = contours.get_frame_points('AORTA_VALVE',
                                                     point.sop_instance_uid)
            atrial_extend,_ = contours.get_frame_points('LAX_LA_EXTENT',
                                                        point.sop_instance_uid)
            if len(aorta_points)>0:
                continue
            if len(atrial_extend)>0:
                continue
            if (index+1) % 3 !=0:
                contours.add_point('MITRAL_VALVE',point)
        del contours.points['LAX_LV_EXTENT']

    if 'LAX_LA_EXTENT' in contours.points.keys():
        for index, point in enumerate(contours.get_timeframe_points(
            'LAX_LA_EXTENT', phases)[1]):
            if(index +1)%3 !=0:
                contours.add_point('MITRAL_VALVE', point)
        del contours.points['LAX_LA_EXTENT']

    if 'LAX_RV_EXTENT' in contours.points.keys():
        for index,point in enumerate(contours.get_timeframe_points(
                'LAX_RV_EXTENT',phases)[1]):
            if (index + 1) % 3 != 0:
                contours.add_point('TRICUSPID_VALVE', point)
        del contours.points['LAX_RV_EXTENT']
except:
        err = 'Computing valve landmarks'
        print(
            '\033[1;33;41m  {0}\t{1}\t\t\t{2}'.format(case, 'Fail',err))


cvi_cont = CVI42XML()
cvi_cont.contour = contours

output_gpfile = os.path.join(working_dir,
                             'GPFile_proc.txt')
output_metafile = os.path.join(working_dir,
                               'SliceInfoFile_proc.txt')

cvi_cont.export_contour_points(output_gpfile)
cvi_cont.export_dicom_metadata(output_metafile)


print('Done')