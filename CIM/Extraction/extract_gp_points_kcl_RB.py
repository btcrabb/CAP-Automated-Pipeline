
#Author: Laura Dal Toso 
#Date: 31 May 2022
#Based on work by: Richard Burns
#------------------------------------------------------------
#Use this script to extract guide points from .cvi42 files and to generate the SliceInfoFile
#
#Before running: 
# - check relative paths

#------------------------------------------------------------


from CVI42XML import *
from Contours import *
from mesh_tools.visualization import Figure
import time
import os

working_dir = '.' #edit to your relative path to this data

output_path = working_dir + 'AB_92_VJ_AM_Bio'
dicom_extension = '.txt' #.dcm
plot_contours = False

dcm_path = os.path.join(working_dir,'AB_92_VJ_AM_Bio_con.metadata.txt')
start = time.time()

contour_file = os.path.join(working_dir,'AB_92_VJ_AM_Bio_con.cvi42wsx')
out_contour_file_name = os.path.join(output_path,'GPFile.txt')
out_metadata_file_name = os.path.join(output_path,'SliceInfoFile.txt')

cvi42Contour = CVI42XML(contour_file,dcm_path,dicom_extension,
                    convert_3D=True,log = True)

contour = cvi42Contour.contour
coords = contour.compute_3D_coordinates(timeframe=[])

if plot_contours == True:
    import matplotlib
    cmap = matplotlib.cm.get_cmap('gist_rainbow')

    contours_types = list(contour.points.keys())
    norm = matplotlib.colors.Normalize(vmin=1,vmax=2*len(contours_types))

    time_frame = [1]
    points_to_plot=[]

    for contour_type in contours_types:
        points_to_plot.append(
            cvi42Contour.contour.get_timeframe_points_coordinates(
            contour_type,time_frame)[1])


    cont_fig = Figure('contours')
    for index, points in enumerate(points_to_plot):
        cont_fig.plot_points(contours_types[index],points,
                             color=cmap(norm(2*index))[:3],size=1.5)


cvi42Contour.contour = contour
cvi42Contour.export_contour_points(out_contour_file_name)
cvi42Contour.export_dicom_metadata(out_metadata_file_name)
