from ctypes.wintypes import tagPOINT
import numpy as np
import pandas as pd
import warnings
import os

#local imports
from . import fitting_tools as tools
import re
from .surface_enum import *
from .Frame import *

#LDT 23/02
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from . import visualization as viewer

SAMPLED_CONTOUR_TYPES = [ContourType.LAX_LV_ENDOCARDIAL,
                         ContourType.LAX_LV_EPICARDIAL,
                         ContourType.SAX_LV_ENDOCARDIAL,
                         ContourType.SAX_LV_EPICARDIAL,
                         ContourType.LAX_LA,
                         ContourType.SAX_RV_ENDOCARDIAL,
                         ContourType.LAX_RV_ENDOCARDIAL,
                         ContourType.SAX_RV_EPICARDIAL,
                         ContourType.SAX_RV_EPICARDIAL,
                         ContourType.SAX_RV_FREEWALL,
                         ContourType.LAX_RV_FREEWALL,
                         ContourType.SAX_RV_SEPTUM,
                         ContourType.LAX_RV_SEPTUM,
                         ContourType.SAX_RV_OUTLET]
UNSAMPLED_CONTOUR_TYPES = [ContourType.MITRAL_VALVE,
                           ContourType.TRICUSPID_VALVE,
                           ContourType.AORTA_VALVE,
                           ContourType.PULMONARY_VALVE,
                            ContourType.APEX_POINT,
                            ContourType.RV_INSERT]


##Author : Charlène Mauger, University of Auckland, c.mauger@auckland.ac.nz
class GPDataSet(object):
    """ This class reads a dataset. A DataSet object has the following properties:

    Attributes:
        case: case name
        mitral_centroid: centroid of the 3D contour points labelled as mitral valve
        tricuspid_centroid: centroid of the 3D contour points labelled as tricuspid valve
        aorta_centroid: centroid of the 3D contour points labelled as aortic valve
        pulmonary_centroid: centroid of the 3D contour points labelled as pulmonic valve
        number_of_slice: number of 2D slices
        number_of_time_frame: number of time frames
        points_coordinates: 3D coordinates of the contour points
    """

    def __init__(self, contour_filename, metadata_filename =None,
                 case ='default' , sampling = 1,
                 time_frame_number =None ):
        """ Return a DataSet object. Each point of this dataset is characterized by
        its 3D coordinates ([Evenly_spaced_points[:,0],Evenly_spaced_points[:,1],
        Evenly_spaced_points[:,2]]), the slice it belongs to (slice_number) and the surface
        its belongs to (ContourType)

            Input:
                filename: filename is the file containing the 3D contour points coordinates,
                            labels and time frame (see example GPfile.txt).
                filenameInfo: filename is the file containing dicom info
                            (see example SliceInfoFile.txt).
                case: case number
                time_frame_number: time frame #
        """
        # arrays

        self.points_coordinates = np.empty((0, 3))
        self.contour_type = np.empty((1))
        self.slice_number = np.empty((1))
        self.weights = np.empty((1))
        self.time_frame =None
        self.number_of_slice = 0
        self.frames = {}

        # strings
        self.case = case

        # scalars
        self.time_frame = time_frame_number
        self._read_contour_file(contour_filename,time_frame_number,sampling)
        self._initialize_landmarkds()
        if metadata_filename != None:
            self._read_dicom_metadata(metadata_filename)
        else:
            warnings.warn('Metadata file is not defined ')


        # Mitral points extracted from Circle belong to the slice -1 (I don't know why...).
        # To be able to apply a breath-hold misregistration correction,
        # we need to find which long axis slice each BP point was extracted
        # from to apply the correct shift.
        # If you don't have this problem with your dataset, you can comment it.
        # self.identify_mitral_valve_points()
    
    def _read_contour_file(self, filename, time_frame_number =None, sampling = 1):
        '''add  by A. Mira 02/2020'''
        # column num 3 of my datset is a space
        if not os.path.exists(filename):
            warnings.warn('Contour files does not exist')
            return
        P = []
        slices =[]
        contypes = []
        weights = []
        time_frame =[]
        try:
            data = pd.read_csv(open(filename), sep='\t', header=None, low_memory = False) #LDT 15/11 added low_memory False
            for line_index,line in enumerate(data.values[1:]):

                P.append([float(x) for x in line[:3]])

                slices.append(int(line[4]))
                contypes.append(line[3])

                weights.append(float(line[5]))
                try:
                    time_frame.append(int(float(line[6])))
                except:
                    time_frame.append(time_frame_number)  # data[6]
            P = np.array(P)
            slices = np.array(slices)
            contypes = np.array(contypes)
            weights = np.array(weights)
            time_frame = np.array(time_frame)
            
        except ValueError:
            print("Wrong file format: {0}".format(filename))


        if time_frame_number is not None:
            valid_contour_index = np.array(time_frame == time_frame_number)
            if (np.sum(valid_contour_index) == 0):
                warnings.warn('Wrong time frame number')
                return

            P = P[valid_contour_index, :]
            slices = slices[valid_contour_index]
            contypes = contypes[valid_contour_index]
            weights = weights[valid_contour_index]

        contypes = self._convert_contour_types(contypes)
        # increment contours points which don't need sampling

        valid_contour_index = np.array([x in UNSAMPLED_CONTOUR_TYPES
                                        for x in contypes])

        self.points_coordinates = P[valid_contour_index]
        self.contour_type = contypes[valid_contour_index]
        self.slice_number = slices[valid_contour_index]
        self.weights = weights[valid_contour_index]
        del_index =  list(np.where(valid_contour_index)[0])
        P = np.delete(P,del_index, axis = 0)
        contypes = np.delete(contypes, del_index)
        slices = np.delete(slices, del_index)
        weights = np.delete(weights, del_index)

        self.number_of_slice = len(self.slice_number)  # slice index starting with 0

        self._sample_contours(P, slices, contypes ,weights,sampling) # there are
        # too many
        # points extracted from cvi files.  To reduce computation time,
        # the contours points are sampled

        self.number_of_slice = max(self.slice_number) + 1


    def _sample_contours(self, points, slices, contypes, weights, sample):

        for j in np.unique(slices):  # For slice i, extract evenly
            # spaced point for all type
            for contour_index, contour_type in enumerate(
                    SAMPLED_CONTOUR_TYPES):

                C = points[(contypes == contour_type) & (slices == j), :]
                C_weights = weights[(contypes == contour_type) & (slices == j)]

                if len(C) > 0:
                    # sort the points by euclidean distance from the
                    # previous point

                    Cx_index,Cx = tools.sort_consecutive_points(C)
                    if len(Cx.shape) == 1:
                        Cx = Cx.reshape(1,-1)


                    self.points_coordinates = np.vstack(
                        (self.points_coordinates, Cx[0::sample, :]))
                    self.slice_number = np.hstack(
                        (self.slice_number, [j] * len(Cx[0::sample, :])))
                    self.contour_type = np.hstack((self.contour_type,
                                                   [contour_type] * len(
                                                       Cx[0::sample, :])))
                    self.weights = np.hstack(
                        (self.weights, C_weights[Cx_index[0::sample]]))

    def _initialize_landmarkds(self):
        " add by A.Mira on 01/2020"
        # calc valve centroids

        P = self.points_coordinates
        mitral_index = (self.contour_type == ContourType.MITRAL_VALVE)
        if (np.sum(mitral_index)> 0):
            self.mitral_centroid = P[mitral_index, :].mean(axis=0)

        tricuspid_index = (self.contour_type == ContourType.TRICUSPID_VALVE)
        if np.sum(tricuspid_index)>0:
            self.tricuspid_centroid = P[tricuspid_index, :].mean(axis=0)

        aorta_contour_index = (self.contour_type == ContourType.AORTA_VALVE)
        if np.sum(aorta_contour_index) >0 :
            self.aorta_centroid = P[aorta_contour_index,:].mean(axis=0)

        pulmonary_index = (self.contour_type == ContourType.PULMONARY_VALVE)
        if np.sum(pulmonary_index) >0:
            self.pulmonary_centroid = P[pulmonary_index, :].mean(axis=0)

        apex_index = (self.contour_type == ContourType.APEX_POINT)
        if np.sum(apex_index)> 0:
            self.apex = P[apex_index, :]

            if len(self.apex) > 0:
                self.apex = self.apex[0, :]

    @staticmethod
    def _convert_contour_types(contours):
        " add by A.Mira on 01/2020"
        # convert contours from string type to Contour enumeration
        # type

        new_contours =  np.empty(contours.shape[0], dtype=ContourType)
        for contour_type in ContourType:
            new_contour_index = np.where(contours == \
                                contour_type.value)[0]
            new_contours[new_contour_index] = contour_type
        return new_contours


    def create_rv_epicardium(self, rv_thickness):
        """ This function generates phantom points for the RV epicardium.
        Epicardium of the RV free wall was not manually contoured in our dataset,
         but it is better to have them when customizing the surface mesh.
        RV epicardial phantom points are estimated by extending the RV endocardium
        contour points by a fixed distance (3mm from the literature).
        If your dataset contains RV epicardial point, you can comment this function
        Input:
            rv_thickness : thickness of the wall to be created
        Output:
            rv_epi: RV epicardial phantom points
        """

        # RV_wall_thickness: normal value from literature
        rv_epi = []
        rv_epi_slice =[]
        rv_epi_contour =[]
        valid_contours = [ContourType.SAX_RV_FREEWALL,
                          ContourType.SAX_RV_OUTLET,
                          ContourType.LAX_RV_FREEWALL]
        epi_contours = [ContourType.SAX_RV_EPICARDIAL,
                        ContourType.LAX_RV_EPICARDIAL]

        for i in np.unique(self.slice_number):

            # For each slice, find centroid cloud point RV_FREEWALL
            # Get contour points

            valid_index = ([x in valid_contours[:2] for x in
                             self.contour_type]) * (self.slice_number == i)
            points_slice = self.points_coordinates[valid_index  , :]

            if len(points_slice) > 0:
                slice_centroid = points_slice.mean(axis=0)
                contour_index =0
            else:
                points_slice = self.points_coordinates[(self.contour_type ==
                                valid_contours[2])
                                    & (self.slice_number == i),:]
                if len(points_slice) > 0:
                    slice_centroid = points_slice.mean(axis=0)
                    contour_index =1
                else:
                    continue

            for j in points_slice:
                # get direction
                direction = j[0:3] - slice_centroid
                direction = direction / np.linalg.norm(direction)
                # Move j along direction by rv_thickness
                new_position = np.add(j[0:3],
                                      np.array([rv_thickness * direction[0],
                                                rv_thickness * direction[1],
                                                rv_thickness * direction[2]
                                                ]))
                rv_epi.append(np.asarray([new_position[0], new_position[1],
                                          new_position[2]]))
                rv_epi_slice.append(i)
                rv_epi_contour.append(epi_contours[contour_index])

        self.add_data_points(np.asarray(rv_epi), np.array(rv_epi_contour),
                             np.array(
                                 rv_epi_slice),
                                 [1] * len(rv_epi))

        return np.asarray(rv_epi),np.array(rv_epi_contour),np.array(
            rv_epi_slice)

    def _read_dicom_metadata(self, name):
        """ This function reads the 'name' file containing dicom info
            (see example SliceInfo.txt).
            Input:
                name: file_name

            Output:
                ImagePositionPatient: ImagePositionPatient attribute (x, y, and z coordinates of
                the upper left hand corner of the image)
                ImageOrientationPatient: ImageOrientationPatient attribute (specifies the direction
                 cosines of the first row and the first column with respect to the patient)
                slice_num: slice #
                PixelSpacing: distance between the center of each pixel
        """
        #  it using csv read
        if not os.path.exists(name):
            return
        lines = []
        with open(name, 'rt') as in_file:
            for line in in_file:
                lines.append(re.split('\s+',line))

        try:
            index_imPos = np.where(['ImagePositionPatient' in x for x in lines[
                0]])[0][0]+1
            index_imOr = np.where(['ImageOrientationPatient' in x for x in lines[
                0]])[0][0]+1
            index_image_id = np.where(['frameID' in x for x in lines[
                0]])[0][0]+1
            index_pixel_spacing = np.where(['PixelSpacing' in x for x in lines[
                0]])[0][0]+1
        except:

            index_imPos = 5
            index_imOr = 9
            index_pixel_spacing =16
            index_image_id = 3
        # lines = lines[1:]

        contour_frames = np.unique(self.slice_number)
        frames_to_use = np.where([ int(line[index_image_id]) in contour_frames
                        for line in lines] )[0]

        all_positions = []
        for i in frames_to_use:
            frame_id = int(lines[i][index_image_id])
            position = np.array(lines[i][index_imPos:index_imPos+3]).astype(
                float)
            all_positions.append(position)
            orientation = np.array(lines[i][index_imOr:index_imOr+6]).astype(
                float)

            spacing =  np.array(lines[i][
                                index_pixel_spacing:index_pixel_spacing+2]
                                ).astype(float)

            new_frame = Frame(frame_id,[float(x) for x in position],
                                 [float(x) for x in orientation],
                                 [float(x) for x in spacing])
            # as only one timeframe is used here (ED or ES) the frame number
            # can be used as an unique frame id
            self.frames.update({frame_id:new_frame})

        unique_positions = sorted(np.unique(all_positions,axis =0),key=lambda x: -x[0])
        for frame_uid in self.frames.keys():
            frame_position = np.array(self.frames[frame_uid].position)
            slice = np.where((unique_positions
                                     ==frame_position).all(axis=1))[0][0]
            self.frames[frame_uid].slice = slice

    def add_data_points(self, points, contour_type, slice_number, weights ):
        """
        add new contour points to a data set
        input:
            points: nx3 array with points coordinates
            contour_type: n list with the contour type for each point
            slice_number: n list with the slice number for each point
        """
        if len(points) == len(contour_type) == len(slice_number) == len(weights):
            self.points_coordinates = np.vstack((self.points_coordinates, points))
            self.slice_number = np.hstack((self.slice_number, slice_number))
            self.contour_type = np.hstack((self.contour_type,contour_type))
            self.weights = np.hstack((self.weights, weights))
        else:
            print('In add_data_point input vectors should have the same lenght')

    def identify_mitral_valve_points(self):
        """ This function matches each Mitral valve point with the LAX slice it
        was  extracted
        from.
            Input:
                None
            Output:
                None. slice_number for each Mitral valve point is changed to
                the corresponding LAX slice number
        """

        mitral_points = self.points_coordinates[(self.contour_type ==
                                                 ContourType.MITRAL_VALVE), :]
        # Extraction code extracts BP points' slice as -1
        idx = self.contour_type == ContourType.MITRAL_VALVE

        new_mitral_points = np.zeros((len(mitral_points), 3))
        corresponding_slice = []
        it = 0
        for slice_id in np.unique(self.slice_number):

            LAX = self.points_coordinates[
                  (self.contour_type == ContourType.LAX_LA)
                  & (self.slice_number == slice_id), :]
            if len(LAX) > 0:
                minimum = 100
                Corr = np.zeros((len(mitral_points), 1))
                NP = np.zeros((len(mitral_points), 3))
                Sl = np.zeros((len(mitral_points), 1))
                # Find corresponding BP on this slices - should be two

                for points in range(len(mitral_points)):
                    i = (np.square(mitral_points[points, :] - LAX)).sum(1).argmin()
                    Corr[points] = np.linalg.norm(
                        LAX[i, :] - mitral_points[points, :])
                    NP[points] = LAX[i, :]
                    Sl[points] = slice_id

                index = Corr.argmin()
                new_mitral_points[it, :] = NP[index, :]
                corresponding_slice.append(float(Sl[index]))

                NP = np.delete(NP, index, 0)
                Sl = np.delete(Sl, index, 0)
                Corr = np.delete(Corr, index, 0)
                it = it + 1

                index = Corr.argmin()
                new_mitral_points[it, :] = NP[index, :]
                corresponding_slice.append(float(Sl[index]))
                it = it + 1

        indexes = np.where((self.contour_type == ContourType.MITRAL_VALVE))
        self.points_coordinates[indexes] =  new_mitral_points
        self.contour_type[indexes] = [ContourType.MITRAL_VALVE] * len(new_mitral_points)
        self.slice_number[indexes] = corresponding_slice

    def create_valve_phantom_points(self, n, contour_type):
        """ This function creates mitral phantom points by fitting a circle to the mitral points
        from the DataSet
            Input:
                n: number of phantom points we want to create
            Output:
                P_fitcircle: phantom points
        """
        new_points=[]
        valid_contour_types = np.array([ContourType.TRICUSPID_VALVE,
                                        ContourType.MITRAL_VALVE,
                                        ContourType.PULMONARY_VALVE,
                                        ContourType.AORTA_VALVE])
        associated_contours = np.array([ContourType.SAX_RV_FREEWALL,
                                        ContourType.SAX_LV_ENDOCARDIAL,
                                        ContourType.SAX_RV_FREEWALL,
                                        ContourType.SAX_LV_ENDOCARDIAL])
        
        if not (contour_type in valid_contour_types):
            return []

        valve_points = self.points_coordinates[self.contour_type ==
                                            contour_type, :]

        if len(valve_points) > n:# if we have enough points to define the
        # contour,
            # better keep the contour itself
            return valve_points
        if len(valve_points) == 0:
            return  new_points

        # Coordinates of the 3D points
        P = np.array(valve_points)

        distance = [np.linalg.norm(P[i]-P[j]) for i in range(len(P)) for j
                    in range(len(P))]

        #if max(distance) < 5:
        if len(distance) ==0 : #(from RB's version)
            # if the distance between points is smaller than 5 mm
            # it will be considered as coincident points
            return np.empty((0,3))
        # in case there are only two points
        #  the valve points will be created using an cercle
        # in case of 3 and more an elipsoid will be fitted.
        valid_points = False
        if max(distance) > 10 and len(valve_points)< 3: #(R.B)
        #if len(valve_points) < 3: (A.M)
            valid_points = True

        

        if valid_points:

            # select just the last slice of the associated contour
            slice_nb = self.slice_number[
                self.contour_type ==   contour_type][0]

            v1 = self.frames[slice_nb].orientation[0:3]
            v2 = self.frames[slice_nb].orientation[3:6]
            normal = np.cross(v1,v2)

            index_valid_pair = np.argmax(distance)

            control_point1 = P[int(np.floor(index_valid_pair / len(P))), :]
            control_point2 = P[index_valid_pair % len(P), :]
            r = max(distance) / 2

            P_mean = 0.5 * (control_point1 + control_point2)
            u = (control_point1 - control_point2)/\
                np.linalg.norm(control_point1 - control_point2)
            # angle1 = np.arccos(np.dot(u, v1))
            # angle2 =np.arccos(np.dot(u,v2))
            # if just two pints are used sometimes the valve are
            # perpendicular to the
            # RV/LV slice
            # In this case use V2 to comput the valve normal
            normal_valve = np.cross(normal, u) #if angle1 > angle2 \
                #else np.cross(v2, u)
            if contour_type == ContourType.PULMONARY_VALVE:
                normal_valve = np.cross(v1,v2)
            t = np.linspace(-np.pi, np.pi, n)
            new_points = tools.generate_2Delipse_by_vectors(t, [0,0], r)


        else:
            P_mean = P.mean(axis=0)
            P_centered = P - P_mean
            U, s, V = np.linalg.svd(P_centered)

            # Normal vector of fitting plane is given by 3rd column in V
            # Note linalg.svd returns V^T, so we need to select 3rd row from V^T
            normal_valve = V[2, :]

            # -------------------------------------------------------------------------------
            # (2) Project points to coords X-Y in 2D plane
            # -------------------------------------------------------------------------------
            P_xy = tools.rodrigues_rot(P_centered, normal_valve, [0, 0, 1])
            #center, r = tools.fit_circle_2d(P_xy[:, 0], P_xy[:, 1])
            center, axis_l,rotation = tools.fit_elipse_2d(P_xy[:,:2])
            #C = np.array([center[0], center[1], 0]) + P_mean

            # --- Generate points for fitting circle
            t = np.linspace(-np.pi, np.pi, n)
            new_points = tools.generate_2Delipse_by_vectors(t, center, axis_l,
                                                     rotation)

        new_points = np.array([new_points[:, 0], new_points[:, 1],
                               [0] * new_points.shape[0]]).T
        new_points = tools.rodrigues_rot(new_points,[0,0,1],
                                         normal_valve)+ P_mean

        # insert new points in the dataset
        # output type depnes on the input contour type
        # and weight are computed as mean weight of the valve points
        if len(new_points) >0:
            if contour_type == ContourType.MITRAL_VALVE:
                output_type = ContourType.MITRAL_PHANTOM
            elif contour_type == ContourType.TRICUSPID_VALVE:
                output_type = ContourType.TRICUSPID_PHANTOM
            elif contour_type == ContourType.AORTA_VALVE:
                output_type = ContourType.AORTA_PHANTOM
            elif contour_type == ContourType.PULMONARY_VALVE:
                output_type = ContourType.PULMONARY_PHANTOM


            weight_MV = self.weights[self.contour_type == contour_type].mean()

            self.add_data_points(new_points,
                                [output_type] * len(new_points),
                                [-1] * len(new_points),
                                [weight_MV] * len(new_points))

        return new_points

    def PlotDataSet(self, contours_to_plot =[]):
        """ This function plots this entire dataset.
            Input:
                Con
            Output:
                traces for figure
        """
        # contours lines types
        contour_lines = np.array([ContourType.TRICUSPID_PHANTOM,
                                  ContourType.LAX_RA,
                                  ContourType.SAX_RV_FREEWALL,
                                  ContourType.LAX_RV_FREEWALL,
                                  ContourType.SAX_RV_SEPTUM,
                                  ContourType.LAX_RV_SEPTUM,
                                  ContourType.SAX_RV_EPICARDIAL,
                                  ContourType.LAX_RV_EPICARDIAL,
                                  ContourType.LAX_RV_ENDOCARDIAL,
                                  ContourType.SAX_RV_ENDOCARDIAL,
                                  ContourType.SAX_RV_OUTLET,
                                  ContourType.PULMONARY_PHANTOM,

                                  ContourType.MITRAL_PHANTOM,
                                  ContourType.LAX_LA,
                                  ContourType.SAX_LV_EPICARDIAL,
                                  ContourType.LAX_LV_EPICARDIAL,
                                  ContourType.SAX_LV_ENDOCARDIAL,
                                  ContourType.LAX_LV_ENDOCARDIAL,
                                  ])
        lines_color_map = np.array(["rgb(128,0,128)","rgb(186,85,211)",
                                    "rgb(0,0,205)", "rgb(65,105,225)",
                                    "rgb(139,0,139)",  "rgb(153,50,204)",
                                    "rgb(0,191,255)", "rgb(30,144,255)",
                                    "rgb(0,0,205)", "rgb(65,105,225)",
                                    "rgb(0,206,209)","rgb(95,158,160)",
                                    "rgb(128,0,0)", "rgb(205,92,92)",
                                    "rgb(220,20,60)","rgb(255,127,80)",
                                    "rgb(85,107,47)","rgb(50,205,50)"])
        # points types
        contour_points = np.array([ContourType.RV_INSERT, ContourType.APEX_POINT,
                                   ContourType.MITRAL_VALVE,
                                   ContourType.TRICUSPID_VALVE,
                                   ContourType.AORTA_VALVE,
                                   ContourType.PULMONARY_VALVE])
        points_color_map = np.array(["rgb(255,20,147)", "rgb(0,191,255)",
                                     "rgb(255,0,0)","rgb(128,0,128)",
                                     "rgb(0,255,0)", "rgb(0,43,0)",])

        if not isinstance(contours_to_plot, list):
            contours_to_plot = [contours_to_plot]
        if len(contours_to_plot) == 0:
            contours_to_plot = contour_lines + contour_points

        contourPlots =[]
        for  contour in contours_to_plot:
            contour_index = np.where(contour_lines == contour)[0]
            points_size =2

            if len(contour_index)== 0:
                contour_index = np.where(contour_points == contour)[0]
                points_size = 5
                if len(contour_index)==1:
                    points_color= points_color_map[contour_index][0]
            else:
                points_color = lines_color_map[contour_index][0]

            if len(contour_index)>0:

                contourPlots = contourPlots + tools.Plot3DPoint(
                    self.points_coordinates[
                                np.where(np.asarray(
                                    self.contour_type) == contour)],
                            points_color, points_size, contour.value)

        return contourPlots


    def sinclaire_slice_shifting(self, frame_num,fix_LAX = False):
        """ This method does a breath-hold misregistration correction be default for both LAX
            and SAX using Sinclair, Matthew, et al. "Fully automated segmentation-based
            respiratory motion correction of multiplanar cardiac magnetic resonance images
            for large-scale datasets." International Conference on Medical Image Computing
            and Computer-Assisted Intervention. Springer, Cham, 2017. Briefly, this
            method iteratively registers each slice to its intersection with the other slices,
             which are kept fixed.
            This is performed at ED only. Translations from ED are applied to the others.
            Rotations and out-of-plane
            displacements were assumed to be negligible relative to the slice thickness.
            If one need to correct the shift for the SAX axis only, set fix_LAX to True
            Input:
               fix_LAX: bool, true if one need to keep LAX contours fixed.
            Output:
               2D translations needed (N*2, where N is the number of slices).
               3D position of the translation (Nx3, where N is the number of slice)
        """

        if ContourType.LAX_LV_EPICARDIAL not in self.contour_type:
            warnings.warn('LAX_LV_EPICARDIAL contour is missing. Slice shift have not been '
                          'corrected')
            return [],[]

        stoping_criterion = 5;  # The stoping_criterion is the residual translation
        # read the slice number corresponding to each frame
        translation = np.zeros((len(self.frames.keys()), 2))  # 2D translation
        position = np.zeros((len(self.frames.keys()), 3))
        iteration_num = 1

        while stoping_criterion > 1 and iteration_num < 50:
            #print('iteration',iteration_num )
            nb_translations = 0
            np_frames = np.unique(self.slice_number)

            int_t = []
            for index, id in enumerate(np_frames):

                #print('slice', id)
                t = self._get_slice_shift_sinclaire(id, iteration_num, fix_LAX)
                # LDT TEST
                #print('iteration: ', iteration_num, '  slice', id, '  shift ', t)
                
                if not (t is None):

                    nb_translations += 1
                    int_t.append(np.linalg.norm(t))

                    translation[index,:] = translation[index,:] + t
                    position[index,:] = self.frames[id].position
                    # the translation is done in 2D
                    point_2_translate = self.points_coordinates[self.slice_number == id,:]
                    transformation = self.get_affine_matrix_from_metadata(id,
                                                                          scaling=False)
                    # the translation is done in 2D
                    P2D_LV = tools.apply_affine_to_points(np.linalg.inv(
                        transformation), point_2_translate)[:,:2]
                    LV = P2D_LV + t

                    # Back to 3D
                    pts_LV = np.zeros((len(LV),3))
                    pts_LV[:,0:2] = LV

                    P3_LV = tools.apply_affine_to_points(transformation,pts_LV)
                    indexes = np.where((self.slice_number == id))
                    self.points_coordinates[indexes,:] = P3_LV

            
            iteration_num = iteration_num + 1
            stoping_criterion =np.max(int_t)

        # update mitral, tricuspid points and apex
        self._initialize_landmarkds()
    

        return translation, position
    
    def get_unintersected_slices(self):
        ## find redundant slices. by Lee
        
        lax_registered_contours = [
                                    ContourType.LAX_LV_ENDOCARDIAL,
                                    ContourType.LAX_RV_FREEWALL,
                                    ContourType.LAX_RV_SEPTUM,
                                    ContourType.LAX_LV_EPICARDIAL
                                   ]
        
        np_slices = np.unique(self.slice_number)
        redundant_slices = []
        for index, id in enumerate(np_slices):
            zero_count = 0
            for c_indx, contour in enumerate(lax_registered_contours):
                contour_intersect_points = \
                    self._get_slice_intersection_points(contour, id)
                if len(contour_intersect_points) == 0:
                    zero_count += 1
                    #print(contour)
            #print('zero_count ', zero_count)
            if zero_count==4:
                redundant_slices.append(id)
                
        if len(self.points_coordinates)==0:
            return np.zeros(0), np.zeros(0)
                
        valid_index = np.ones(self.contour_type.shape, dtype=bool)
        for i in redundant_slices:
            valid_index = valid_index * (self.slice_number != i)

        #remove unitersetcted sax slices (based on LAX contours), 
        self.points_coordinates = self.points_coordinates[valid_index]
        self.contour_type = self.contour_type[valid_index]
        self.slice_number = self.slice_number[valid_index]
        self.weights = self.weights[valid_index]
        
        return redundant_slices, valid_index
    
    def get_unintersected_slices_RV(self):
        ## find redundant slices. by Lee
        
        sax_registered_contours = [
                                    ContourType.SAX_RV_FREEWALL,
                                    ContourType.SAX_RV_SEPTUM,
                                    ContourType.SAX_RV_OUTLET,
                                   ]
        
        lax_registered_contours = ContourType.LAX_RV_SEPTUM
        
        np_slices = np.unique(self.slice_number)
        redundant_slices = []
        for index, id in enumerate(np_slices):
            contour_intersect_points = \
                self._get_slice_intersection_points(lax_registered_contours, id)
            if len(contour_intersect_points) == 0:
                redundant_slices.append(id)
                
        if len(self.points_coordinates)==0:
            return np.zeros(0), np.zeros(0)
                
        invalid_index = np.ones(self.contour_type.shape, dtype=bool)
        for i in redundant_slices:
            for c_indx, contour in enumerate(sax_registered_contours):
                invalid_index = invalid_index * (self.contour_type == contour) \
                                * (self.slice_number == i)
        valid_index = ~invalid_index

        #remove unitersetcted sax slices (based on LAX contours), 
        self.points_coordinates = self.points_coordinates[valid_index]
        self.contour_type = self.contour_type[valid_index]
        self.slice_number = self.slice_number[valid_index]
        self.weights = self.weights[valid_index]
        
        return redundant_slices, valid_index

    #@profile
    def _get_slice_shift_sinclaire(self, slice_number, iter, fix_LAX = False):
        """
            computes translation of slice #slice_number minimizing the
            distance between it's corresponding contour points  and the
            intersection points of the slice plane with the contours from
            any other slice of the stack
            input:
                slice_number : slice number to compute the translation for
            returns:
             t : 2D translation vector, empty array if no intersection points
                have been found
        """

        # ----LDT: define contours to be used
        sax_registered_contours = [ContourType.SAX_LV_ENDOCARDIAL,
                                   ContourType.SAX_RV_FREEWALL,
                                   ContourType.SAX_RV_SEPTUM,
                                   ContourType.SAX_RV_OUTLET,
                                   ContourType.SAX_LV_EPICARDIAL
                                   ]
        lax_registered_contours = [ContourType.LAX_LV_ENDOCARDIAL,
                                   ContourType.LAX_RV_FREEWALL,
                                   ContourType.LAX_RV_SEPTUM,
                                   ContourType.LAX_RV_FREEWALL,
                                   ContourType.LAX_LV_EPICARDIAL
                                   ]

        #associated_surface = [Surface.LV_ENDOCARDIAL]
        p2_reference = []
        intersection_points_2d = []
        #model_intersection_lv = []
        weights = []
        #data_points_lv = []
        
        # -------LDT: transform origin (--------why?)
        origin_transformation = np.linalg.inv(
            self.get_affine_matrix_from_metadata(slice_number,
                                                 scaling=False))


        # the moving contour points need to be projected into the fixed
        # contour points and the model surface,
        # the binoms of contours types and surface types are created
        # for example the SAX LV edocardial contours points will be projected
        # into the LAX LV edocardial contours and the LV edocardial surface of the model
        
        for c_indx, contour in enumerate(sax_registered_contours):
            # -----LDT: check if selected contour is in my GPFile
            # -----LDT: only pick the given slice number ---- what kind of data does a slice contain?
            valid_index = (self.contour_type == contour) \
                          & (self.slice_number == slice_number)
            
            ref_ctype = contour

            if np.any(valid_index):
                # if the contours is a SAX contours the intersecting contour,
                # if found, is in LAX contours

                # -----LDT: extract point coordinates for the GPFile data relative 
                # -----LDT: to the correct slice and contour type (SAX)
                contour_points_ref = self.points_coordinates[valid_index, :]
                # -----LDT: select the corresp. contour from LAX list
                intersecting_contour = lax_registered_contours[c_indx]

            # if there are no points corresponding to the SAX contours (LDT: in the GPFile),
            # check if the slice contains LAX contours and associate
            # with the intersecting SAX contours
            else:
                if not fix_LAX:
                    valid_index = (self.contour_type == lax_registered_contours[c_indx]) \
                                  & (self.slice_number == slice_number)
                    ref_ctype = lax_registered_contours[c_indx]
                    if np.any(valid_index):
                        contour_points_ref = self.points_coordinates[valid_index, :]

                        intersecting_contour = sax_registered_contours[c_indx]

                    else:
                        continue
                else:
                    continue

            # Compute contours intersection with  Dicom slice i.
            # -----LDT: return intersection points of the contour of  "#slice_number"
            # -----LDT: with the contours from  any  other slice in the stack.
            contour_intersect_points = \
                self._get_slice_intersection_points(intersecting_contour,
                                                    slice_number)

            '''
            # -----LDT: plot figure of the points
            
            #if iter == 2:
                
            fig = plt.figure(figsize=(10, 10))
            ax = plt.axes(projection='3d')
            ax.scatter3D(*zip(*contour_points_ref), cmap = 'xkcd:orange', label = "contour_points_ref")

            if len(contour_intersect_points) >=1:
                    ax.scatter3D(*zip(*contour_intersect_points), label = "contour intersect points")

            else:
                    pass

            #if (slice_number == 192) & (contour == ContourType.SAX_LV_EPICARDIAL):
                #print('contour_intersect_points', contour_intersect_points)
            
            ax.legend()
            ax.set_title('inters. points = '+str(len(contour_intersect_points)))
            plt.savefig('./results/Slice'+ str(slice_number)+'_it'+str(
                        iter)+'_'+str(ref_ctype )[12:]+'.png')
                
            #print('----- ref_contour: ', ref_ctype , len(contour_intersect_points))
            '''
            
            if not (len(contour_intersect_points) == 0):

                # LDT: apply affine matrix to 3D points, only in-plane transformation is considered
                # LDT: tansformation is applied both to the reference contour points and to the intersecting contour points
                intersection_points_2d.append(tools.apply_affine_to_points(
                    origin_transformation, contour_intersect_points)[:, :2])

                p2_reference.append(list(tools.apply_affine_to_points(
                    origin_transformation, contour_points_ref)[:, :2]))
                weights.append(1)
            
            # LDT TEST:
            #if (slice_number == 192) & (contour == ContourType.SAX_LV_EPICARDIAL):
                #print('contour', contour)
                #print('intersection_points_2d: ', tools.apply_affine_to_points(
                    #origin_transformation, contour_intersect_points)[:, :2], '  p2_reference: ', list(tools.apply_affine_to_points(
                    #origin_transformation, contour_points_ref)[:, :2]))

        # Find closest points
        # Multidimensional unconstrained nonlinear minimization (Nelder-Mead).
        # Starts at X0 = [0,0] and attempts to find a local minimizer
        t =np.array([0,0])
        
        if len(intersection_points_2d) > 0:
            if not (len(intersection_points_2d) == 1 and len(intersection_points_2d[0])==1):
                # LDT:  compute the optimal translation between two sets of grouped points
                t= -tools.register_group_points_translation_only(
                    intersection_points_2d,p2_reference, slice_number, norm=1)
                


        return t


    def _get_slice_intersection_points(self,contour, slice_number):
        """  return intersection points of the contour of  "#slice_number"
        with the contours from  any  other slice in the stack. There the
        points of a slice are assumed to be ordered also in slice position is considered !!
            input:
                slice_number : slice index to compute the intersections
            output: nx2 array- coordinates of the intersection points,
                    empty array if none
        """

        intersection_points = np.empty((0, 3))
        slices = [x for x in np.unique(self.slice_number) if x !=slice_number]
        for j in slices:

            j_valid_index = (self.contour_type == contour) \
                               * (self.slice_number == j)


            if np.sum(j_valid_index)>2:
                lv_epi = self.points_coordinates[j_valid_index, :]
                _,lv_epi =tools.sort_consecutive_points(lv_epi)

            else:
                continue

            for o in range(len(lv_epi) - 1):
                #  some contour are not closed contours
                # like ethe free wall
                # therefore if the points are too fat we need to exclude them
                P =[]
                if np.linalg.norm(lv_epi[o+1]-lv_epi[o-1])<10:
                    P = tools.LineIntersection(
                        self.frames[slice_number].position,
                        self.frames[slice_number].orientation,
                        lv_epi[o, :], lv_epi[o + 1, :])
                
                #if 192 not in slices:
                    #print('P', P)
                # check the condition to have an intersection + the
                # intersection point is in between the two defined points
                if len(P) > 0 and np.dot(P.T - lv_epi[o, :],\
                                         P.T - lv_epi[o + 1, :]) < 0:

                    intersection_points = np.vstack((intersection_points, P))

        return intersection_points

    def get_affine_matrix_from_metadata(self, slice_num, scaling = True):
        """ This function calculates affine matrix describing the slice
        position in the space coordinate system, given slice number.

            Input:
                slice_num: slice #

            Output:
                T: affine matrix
        """
        T = self.frames[slice_num].get_affine_matrix(scaling=scaling)


        return T

    def apply_slice_shift(self, frames_translation, position, frame_uids = None):
        """ This function applies 2D translations from breath-hold
        misregistration correction to the DataSet .

            Input:
                translation: list of translations to apply at each corresponding
                            frame (output from LVLAXSAXSliceShifting),
                            the tramslation index corresponds to the frame slice
                            not to the frame id!!!!
                frame_uids: list of frame uid to apply translation,
                            if frames are specified the frame idex should coincide
                            with translation index
            Output:
                None. The Dataset 'self.DataSet' is translated in the function itself.
        """
        # this method is a copy from BiventricularModel class. A. Mira
        # As the changes are done on the DataSet itself without using the model
        # this is a method of GPDataSet class

        # the loop should be done on a frame, the frames have always the same
        #  patient position
        if frame_uids is None:
            frame_uids = list(self.frames.keys())
        if not isinstance(frame_uids,list):
            frame_uids = [frame_uids]

        # read the frame slice number corresponding to the frame ids
        frame_slice_nb = []
        for uid in frame_uids:
            frame_slice_nb.append(self.frames[uid].slice)
        for index,translation in enumerate(frames_translation):

            for frame in self.frames.values():
                if not(np.all(frame.position == position[index])):
                    continue

                frame_uid = frame.image_id
                frame_points  =  self.points_coordinates[ (self.slice_number ==
                                                   frame_uid), :]
                if len(frame_points) > 0:
                    # Get 2D points
                    transformation = self.get_affine_matrix_from_metadata(frame_uid,
                                                                          scaling=False)
                    # the translation is done in 2D
                    P2D_LV = tools.apply_affine_to_points(np.linalg.inv(
                        transformation), frame_points)[:, :2]

                P2D_LV = P2D_LV + translation

                # Back to 3D
                pts_LV = np.zeros((len(P2D_LV), 3))
                pts_LV[:, 0:2] = P2D_LV

                P3_LV = tools.apply_affine_to_points(transformation, pts_LV)
                indexes = np.where((self.slice_number == frame_uid))
                self.points_coordinates[indexes, :] = P3_LV

    def stiff_model_slice_shifting(self, model):
        ''' Performs breath-hold misregistration correction when the dataset
        does not contain any LAX slices
        a stiff linear least squares fit of the LV_ENDOCARDIAL with D-Affine
        regularisation should be performed to align a 3D LV_ENDOCARDIAL model with the long
        axis (defined by mitral point, apex and tricuspid points) slices.
        By using a very stiff fit, the overall shape is preserved.
        Intersections between the 3D model and the 2D SAX slices are
        then calculated and the 2D short axis (SAX)
            slices are aligned with the intersection contours.
               Input: biventricular model
               Output: 2D Translations, position (3D)'''

        translation = np.zeros((len(self.frames.keys()), 2))  # 2D translation
        position = np.zeros((len(self.frames.keys()), 3))
        visited_frames = []
        # Calculate intersection
        # -----------------------
        np_frames = np.unique(self.slice_number)

        for index, id in enumerate(np_frames):
            # this loop will search to compute, first a displacement based on
            # the distance between the centroid of the edocardial points
            # corresponding to the frame, and the centroid of the intersection
            # points bw endocardial surface(model) and the frame plane.
            # if there are no intersection points. The centroid will be
            # aligned with the centroid of the closest frame
            active_frame = id
            frames_subset= list(self.frames.keys())

            while id not in visited_frames or len(frames_subset) == 0:
                # Get all the points on the slice i
                # ----------------------------------
                lv_edo_points = self.points_coordinates[(
                            self.contour_type == ContourType.SAX_LV_ENDOCARDIAL) &
                             (self.slice_number == active_frame),:]
                        # Check if there is an endocardial contour for the slice i.
                # -----------------------------------------------------------
                if len(lv_edo_points) == 0:
                    visited_frames.append(id)
                    continue

                # Get the transformation from 2D to 3D
                # ----------------------------------------
                transformation = self.get_affine_matrix_from_metadata(active_frame,
                                                                      scaling=False)
                P2_LV = tools.apply_affine_to_points(np.linalg.inv(
                    transformation), lv_edo_points)[:,:2]
                # Give the intersection of the LV_ENDOCARDIAL with the slices
                # -----------------------------------------------
                intersection_surface = model.get_intersection_with_dicom_image(self.frames[active_frame])

                # If intersection
                if len(intersection_surface) > 0 :

                    #go back to the dicom coordinates system
                    intersection_surface_2D = tools.apply_affine_to_points(
                        np.linalg.inv(transformation), intersection_surface)[:,:2]
                    centroid_model = tools.compute_area_weighted_centroid(
                        intersection_surface_2D)


                    # Get centroid
                    centroid_LV_Data = P2_LV.mean(axis=0)
                    displacement = centroid_model - centroid_LV_Data
                    translation[index, :] = translation[index, :] + displacement
                    position[index, :] = self.frames[active_frame].position
                    visited_frames.append(id)

                    # Get all points slice
                    frame_points = self.points_coordinates[(self.slice_number
                                                            == id), :]
                    points_2D = tools.apply_affine_to_points(np.linalg.inv(
                        transformation), frame_points)[:,:2]
                    points_2D = points_2D +displacement

                    # Back to 3D
                    frame_points = np.zeros((len(points_2D), 3))
                    frame_points[:, :2] = points_2D
                    frame_points = tools.apply_affine_to_points(
                        transformation, frame_points)
                    indexes = np.where(self.slice_number == id)
                    self.points_coordinates[indexes] = frame_points
                else :
                    frames_subset.remove(id)
                    active_frame = self.get_frame_neighbour(id, frames_subset)

        return translation, position

    def combined_slice_shifting(self, model, fix_LA = False):
        """ This method does a breath-hold misregistration correction for both LAX
            and SAX using  a combined version of Sinclair method and the stiff model method.
            Briefly, this  method iteratively registers each slice to its intersection with
                1) the other slices which are kept fixed
                2) the centroid computed using the slice endocardial contour
                 with the centroid computed using the intersection points between the slice
                 an the model

            Input:
               model :  motel to be used as reference for shift
            Output:
               translation : in plane translation (2D list) odf each slice.
               position: slice position (3D list)
        """

        if ContourType.LAX_LV_EPICARDIAL not in self.contour_type:
            warnings.warn('LA contour is missing. Slice shift have not been '
                          'corrected')
            return [],[]
        tol = 5  # The stoping_criterion is the residual translation
        # read the slice number corresponding to each frame
        translation = np.zeros((len(self.frames.keys()), 2))  # 2D translation
        position = np.zeros((len(self.frames.keys()), 3))
        iteration_num = 1

        while tol > 1 and iteration_num < 100:

            nb_translations = 0
            nb_frames = np.unique(self.slice_number)
            int_t = []
            for index, id in enumerate(nb_frames):

                t = self._get_slice_shift_combined(id, model, fix_LA)
                if (t is not None):

                    nb_translations += 1
                    int_t.append(np.linalg.norm(t))

                    translation[index,:] = translation[index,:] + t
                    position[index,:] = self.frames[id].position
                    # the translation is done in 2D
                    point_2_translate = self.points_coordinates[self.slice_number == id,:]
                    transformation = self.get_affine_matrix_from_metadata(id,
                                                                          scaling=False)
                    # the translation is done in 2D
                    P2D_LV = tools.apply_affine_to_points(np.linalg.inv(
                        transformation), point_2_translate)[:,:2]
                    LV = P2D_LV + t

                    # Back to 3D
                    pts_LV = np.zeros((len(LV),3))
                    pts_LV[:,0:2] = LV


                    P3_LV = tools.apply_affine_to_points(transformation,pts_LV)
                    indexes = np.where((self.slice_number == id))
                    self.points_coordinates[indexes,:] = P3_LV

            iteration_num = iteration_num+1
            if len(int_t)>0:
                tol =np.max(int_t)

        # update mitral, tricuspid points and apex
        self._initialize_landmarkds()

        return translation, position

    def _get_slice_shift_combined(self, slice_number, model,
                                  fix_LA = False):
        """
            computes translation of slice #slice_number minimizing the
            distance between
            1. it's corresponding contour points  and the
            intersection points of the slice plane with the contours from
            any other slice of the stuck
            2. in-slice centroid of the endocardial contour and
            the centroid of the
            input:
                slice_number : slice number to compute the translation for
                model :  reference model
            returns:
             t : 2D translation vector, empty array if no intersection points
                have been found
        """
        # The contours to use for slice shift
        sax_registered_contours = [ContourType.SAX_LV_ENDOCARDIAL,
                                  ContourType.SAX_RV_FREEWALL,
                                   ContourType.SAX_RV_SEPTUM,
                                  ContourType.SAX_RV_OUTLET,
                                  ContourType.SAX_LV_EPICARDIAL
                                   ]
        # additional contours to use for slice shift in case LAX is not fixed
        lax_registered_contours = [ContourType.LAX_LV_ENDOCARDIAL,
                                  ContourType.LAX_RV_FREEWALL,
                                   ContourType.LAX_RV_SEPTUM,
                                  ContourType.LAX_RV_FREEWALL,
                                   ContourType.LAX_LV_EPICARDIAL
                                   ]
        associated_surface = [Surface.LV_ENDOCARDIAL]
        p2_reference = []
        intersection_points_2d = []
        model_intersection_lv = []
        weights = []
        data_points_lv = []

        origin_transformation = np.linalg.inv(
            self.get_affine_matrix_from_metadata(slice_number,
                                                 scaling=False))

        # the moving contour points need to be projected into the fixed
        # contour points and the model surface,
        # the binoms of contours types and surface types are created
        # for example the SAX LV endocardial contours points will be projected
        # into the LAX LV endocardial contours and the LV endocardial surface of the model
        for c_indx,contour in enumerate(sax_registered_contours):
            valid_index = (self.contour_type == contour)\
                             & (self.slice_number == slice_number)

            if np.any(valid_index):
                #if the contours is a SAX contours the intersecting contour,
                # if found, is in LAX contours
                contour_points_ref = self.points_coordinates[valid_index, :]

                intersecting_contour = lax_registered_contours[c_indx]
            # if there are no points corresponding to the SAX contours,
            # check if the slice contains LAX contours and associate
            # with the intersecting SAX contours
            else:
                if not fix_LA:
                    valid_index = (self.contour_type == lax_registered_contours[c_indx]) \
                                     & (self.slice_number == slice_number)
                    if np.any(valid_index):
                        contour_points_ref = self.points_coordinates[valid_index, :]

                        intersecting_contour = sax_registered_contours[c_indx]
                    else:
                        continue
                else:
                    continue

            # Compute contours intersection with  Dicom slice i
            contour_intersect_points = \
                self._get_slice_intersection_points(intersecting_contour,
                                                    slice_number)

            # the lv endocardial points will be used to compute
            # the endocardial centroid and align it with the
            # centroid of the points given by slice intersection with the
            #  endocardial surface of the model
            if contour == ContourType.SAX_LV_ENDOCARDIAL:
                data_points_lv = data_points_lv + list(contour_points_ref)

            # Compute intersection with model
                model_intersection_lv = model_intersection_lv + \
                    list(model.get_intersection_with_dicom_image(
                    self.frames[slice_number], [associated_surface[c_indx]]))

            if not (len(contour_intersect_points) == 0 ):
                #The transformationis done in 2D is
                intersection_points_2d.append(tools.apply_affine_to_points(
                    origin_transformation,contour_intersect_points)[:, :2])
                p2_reference.append(list(tools.apply_affine_to_points(
                    origin_transformation, contour_points_ref)[:, :2]))
                weights.append(1)



        nb_groups = len(intersection_points_2d)
        if len(model_intersection_lv) > 0 and len(data_points_lv)>0:

            # exclude centroid if the intersection with the model is an open contour
            _, model_intersection_lv = \
                tools.sort_consecutive_points(model_intersection_lv)
            inter_distance = \
                np.sum(np.abs(model_intersection_lv -
                       np.roll(model_intersection_lv, 1, axis=0)
                       ) ** 2, axis=1) ** (1. / 2)
            # if open contour then max(inte_distamce) > 10
            if max(inter_distance) < 10:
                # add centroid of LV endo as intersection with the model and the
                # the LV endo centroid from data as a group to be registered
                model_intersection_2d = tools.apply_affine_to_points(
                        origin_transformation,model_intersection_lv)[:, :2]
                model_2d_centroid =  np.mean(model_intersection_2d,axis = 0)

                data_points_lv_2d = tools.apply_affine_to_points(origin_transformation,
                                                              data_points_lv)[:, :2]
                data_2D_centroid = np.mean(data_points_lv_2d, axis = 0)
                if nb_groups ==0:
                    weights.append(1)
                else:
                    weights.append(nb_groups)
                p2_reference.append([data_2D_centroid])
                intersection_points_2d.append([model_2d_centroid])


        t = np.array([0,0])
        # chech for how many intersetion points have been found
        # if just one point for just two contours the registration is not
        # performed
        nb_points = np.sum([len(p)>1 for p in intersection_points_2d])
        if len(intersection_points_2d) > 0 :
            if not (len(intersection_points_2d) == 1 and len(intersection_points_2d[0]) == 1):

                t= -tools.register_group_points_translation_only(
                intersection_points_2d,p2_reference, weights,
                    exclude_outliers=False, norm=2)


        return t

    def get_frame_neighbour(self, frame_id, frame_subset =[]):
        if not isinstance(frame_subset,list):
            frame_subset = [frame_subset]
        if len(frame_subset) ==0:
            frame_subset = self.frames.keys()


        pos_z = self.frames[frame_id].position[2]
        distance = [pos_z - self.frames[new_frame].position[2] for new_frame in
                    frame_subset]
        return frame_subset[np.argmin(distance)]

