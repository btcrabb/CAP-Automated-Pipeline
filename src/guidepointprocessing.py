#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import cv2


def write_to_gp_file(path, coords, label, slice_id, weight=1.0, phase=1.0):

    """ Writes a coordinate/line to the guide point file """
    
    # check if output exists
    if os.path.exists(path):
        flag = 'a'
    else:
        flag = 'w'
        
    with open(os.path.join(path), flag) as f:
        if flag == 'w':
            f.write('x\ty\tz\tcontour type\tframeID\tweight\ttime frame\n')
        for coord in coords:
            f.write('{:.5f}\t{:.5f}\t{:.5f}\t{}\t{}\t{}\t{}\n'.format(coord[0], coord[1], coord[2],
                                                          label, slice_id, weight, phase))

def inverse_coordinate_transformation(coordinate, imagePositionPatient, imageOrientationPatient, ps):

    """ Performs a coordinate transformation from image coordinates to patient coordinates """

    # image position and orientation
    S = imagePositionPatient
    X = imageOrientationPatient[:3]
    Y = imageOrientationPatient[3:]
    
    # construct affine transform
    M = np.asarray([[X[0]*ps[0], Y[0]*ps[1], 0, S[0]],
                [X[1]*ps[0], Y[1]*ps[1], 0, S[1]],
                [X[2]*ps[0], Y[2]*ps[1], 0, S[2]],
                [0, 0, 0, 1]])
    
    # expand dimensions of coordinate
    pos = [float(x) for x in coordinate[0:3]]
    coord = [pos[1], pos[0], 0, 1.0]
    
    # perform transformation and return as list
    return [np.round(x,5) for x in M @ coord]

def remove_valve_points(point_list, landmarks):
    
    """ Removes points that are inside of two landmarks (used to remove points in valve planes) """
    
    # find centroid of landmarks
    p1 = landmarks[0]
    p2 = landmarks[1]
    centroid = find_center(p1, p2)
    radius = int(np.sqrt( ( ( p1[1] - centroid[0] ) ** 2 ) + 
                           ( ( p1[0] - centroid[1] ) ** 2 ) ))

    # empty array placeholder
    placeholder = np.zeros([256,256])
    mask = cv2.circle(placeholder, centroid, radius, 255, -1)
    
    # find points to remove
    mask_list = np.array(np.where(mask > 0))
    
    temp = []
    for point in point_list:
        if point[0] in mask_list[0] and point[1] in mask_list[1]:
            pass
        else:
            temp.append(point.tolist())

    return np.array(temp, dtype=np.int64)

def find_center(p1, p2):
    
    """ finds the centroid of two points """

    return [ int((p1[1] + p2[1])/2), int((p1[0] + p2[0])/2 )]


class GuidePointProcessing():

    # Class Instance that writes guide points to file from image segmentations

    # Class Variables

    # init method
    def __init__( self, patient,
                image_folder,
                segmentation_folder,
                output_folder,
                slice_info_df,
                landmark_dataframe ):

            # Instance Variables
            self.patient = patient                                  # (str) the patient ID
            self.image_folder = image_folder                        # (str) directory of input images
            self.segmentation_folder = segmentation_folder          # (str) directory of segmentations
            self.output_folder = output_folder                      # (str) diretory to save GP files
            self.slice_info_df = slice_info_df                      # (DateFrame) DF of slice information
            self.landmarks_df = landmark_dataframe                  # (DataFrame) DF of landmark points (MV/AV/PV valves)
            self.sa_seg_files = []                                  # (list) list of SAX segmentations
            self.la_seg_files = []                                  # (list) list of LAX segmentations


    def get_segmentation_files(self):

        """ Returns available short and long axis segmentations """

        # find seg files
        for (root, subdir, files) in os.walk(self.segmentation_folder):
            for file in files:
                if '.nii.gz' in file:
                    path = os.path.join(root, file)
                    if 'SA' in path:
                        self.sa_seg_files.append(path)
                    else:
                        self.la_seg_files.append(path)

    def get_intersections(self, point_list1, point_list2, distance_cutoff=4.5):

        """ Finds the points that are within a given cutoff distance between two lists """

        a = range(len(point_list1))
        b = range(len(point_list2))
        [A, B] = np.meshgrid(a,b)
        c = np.concatenate([A.T, B.T], axis=0)
        pairs = c.reshape(2,-1).T
        dist = np.sqrt( ( ( point_list1[pairs[:,0],0] - point_list2[pairs[:,1],0] ) ** 2 ) + 
                       ( ( point_list1[pairs[:,0],1] - point_list2[pairs[:,1],1] ) ** 2 ) )
        
        pairs = pairs[np.where(dist < distance_cutoff)[0].tolist()]

        return pairs


    def process_short_axis(self, display=False):

        """ Processes the short axis segmentations """

        # iterate through SAX segmentations
        for file in self.sa_seg_files:

            # change input name to include modality info (nnUnet formatting)
            view, slice_id, time = file.split('\\')[-1].split('_')
            time = int(time.split('.')[0])
            prefix = file.split('\\')[-1].split('.')
            input_name = 'SA\\' + prefix[0]  + '_0000.' + prefix[1] + '.' + prefix[2]
            
            segmentation = nib.load(file).get_fdata().T[0,:,:].T
            image = nib.load(os.path.join(self.image_folder, input_name)).get_fdata().T[0,:,:]

            # load necessary information for coordinate transformation
            _slice_info = self.slice_info_df[self.slice_info_df['Slice ID'] == int(slice_id)]
            S = _slice_info['ImagePositionPatient'].values[0]
            imgOrient = _slice_info['ImageOrientationPatient'].values[0]
            ps = _slice_info['Pixel Spacing'].values[0]

            # extract points
            LV_endo = (segmentation == 1).astype(np.uint8)
            LV_myo = (segmentation == 2).astype(np.uint8)
            LV_epi = (LV_endo | LV_myo).astype(np.uint8)
            RV_endo = (segmentation == 3).astype(np.uint8)
            RV_myo = (segmentation == 4).astype(np.uint8)
            RV_epi = (RV_endo | RV_myo).astype(np.uint8)

            # convert to contours
            contours, hierarchy = cv2.findContours(cv2.inRange(LV_endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            LV_endo_pts = np.array([x.tolist() for i,x in enumerate(contours[0][:, 0, :]) if i % 2 == 0 ], dtype=np.int64)   
            contours, hierarchy = cv2.findContours(cv2.inRange(LV_epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            LV_epi_pts = np.array([x.tolist() for i,x in enumerate(contours[0][:, 0, :]) if i % 2 == 0 ], dtype=np.int64) 
            contours, hierarchy = cv2.findContours(cv2.inRange(RV_endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            RV_endo_pts = np.array([x.tolist() for i,x in enumerate(contours[0][:, 0, :]) if i % 2 == 0 ], dtype=np.int64) 
            contours, hierarchy = cv2.findContours(cv2.inRange(RV_epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            RV_epi_pts = np.array([x.tolist() for i,x in enumerate(contours[0][:, 0, :]) if i % 2 == 0 ], dtype=np.int64) 

            # Get intersection points between RV endo and LV epi
            if len(RV_endo_pts)>0 and len(LV_epi_pts)>0:
                pairs = self.get_intersections(RV_endo_pts, LV_epi_pts, distance_cutoff=4.5)

                if len(pairs) > 0:
                    RV_septal_pts = RV_endo_pts[np.unique(pairs[:,0])] # deletes intersection from RV endo pts

            # Get intersection points between RV epi and LV epi
            if len(RV_epi_pts)>0 and len(LV_epi_pts)>0:
                
                pairs = self.get_intersections(RV_epi_pts, LV_epi_pts, distance_cutoff=4.5)
                
                if len(pairs) > 0:
                    intersection_points = LV_epi_pts[pairs[:,1],:] # intersection points on LV
                    intersection_on_LV_idx = np.unique(pairs[:,1])
                    LV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(LV_epi_pts) if i not in intersection_on_LV_idx], 
                                          dtype=np.int64)                       
                    RV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_epi_pts) if i not in np.unique(pairs[:,0])], 
                                          dtype=np.int64)

            if display:
                plt.figure(figsize=(12,12))
                plt.imshow(image,cmap='gray')
                plt.scatter(RV_endo_pts[:,1], RV_endo_pts[:,0], s=5, c='#F2CA19')
                plt.scatter(RV_septal_pts[:,1], RV_septal_pts[:,0], s=5, c='#E11845')
                plt.scatter(RV_epi_pts[:,1], RV_epi_pts[:,0], s=5, c='#0057E9')
                plt.scatter(LV_epi_pts[:,1], LV_epi_pts[:,0], s=5, c='#0057E9')
                plt.scatter(LV_endo_pts[:,1], LV_endo_pts[:,0], s=5, c='#87E911')
                plt.show()

            # transform to patient coordinates and write to GP file
            point_lists = [RV_endo_pts,
                          RV_epi_pts,
                          RV_septal_pts,
                          LV_epi_pts,
                          LV_endo_pts]
            labels = ['SAX_RV_FREEWALL',
                     'SAX_RV_EPICARDIAL',
                     'SAX_RV_SEPTUM',
                     'SAX_LV_EPICARDIAL',
                     'SAX_LV_ENDOCARDIAL']

            for i,points in enumerate(point_lists):
                if len(points)>2:
                    pts = [inverse_coordinate_transformation(point, S, imgOrient, ps)
                               for point in points.tolist()]

                    if time > 1:
                        # write to file
                        write_to_gp_file(self.output_folder + '/GP_ES.txt', pts, labels[i], slice_id, weight=1.0, phase=float(time))
                    else:
                        write_to_gp_file(self.output_folder + '/GP_ED.txt', pts, labels[i], slice_id, weight=1.0, phase=float(time))

    def process_long_axis(self, display=False):

        """ Processes the long axis segmentations """

        for file in self.la_seg_files:
    
            # change input name to include modality info (nnUnet formatting)
            view, slice_id, time = file.split('\\')[-1].split('_')
            time = int(time.split('.')[0])
            prefix = file.split('\\')[-1].split('.')
            input_name = '{}\\'.format(view) + prefix[0]  + '_0000.' + prefix[1] + '.' + prefix[2]
            
            segmentation = nib.load(file).get_fdata().T[0,:,:].T
            image = nib.load(os.path.join(self.image_folder, input_name)).get_fdata().T[0,:,:]
            
            if view == 'RVOT':

                # process RVOT views separately as the segmentation labels are different
                # load necessary information for coordinate transformation
                _slice_info = self.slice_info_df[self.slice_info_df['Slice ID'] == int(slice_id)]
                S = _slice_info['ImagePositionPatient'].values[0]
                imgOrient = _slice_info['ImageOrientationPatient'].values[0]
                ps = _slice_info['Pixel Spacing'].values[0]

                # extract points
                RV_endo = (segmentation == 1).astype(np.uint8)
                RV_myo = (segmentation == 2).astype(np.uint8)
                RV_epi = (RV_endo | RV_myo).astype(np.uint8)

                # convert to contours
                contours, hierarchy = cv2.findContours(cv2.inRange(RV_endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                if len(contours) > 0:
                    RV_endo_pts = np.array([x.tolist() for i,x in enumerate(contours[0][:, 0, :]) if i % 2 == 0 ], dtype=np.int64) 
                else:
                    RV_endo_pts = []
                contours, hierarchy = cv2.findContours(cv2.inRange(RV_epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                if len(contours) > 0:
                    RV_epi_pts = np.array([x.tolist() for i,x in enumerate(contours[0][:, 0, :]) if i % 2 == 0 ], dtype=np.int64) 
                else:
                    RV_epi_pts = []
                contours, hierarchy = cv2.findContours(cv2.inRange(RV_myo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                if len(contours) > 0:
                    RV_myo_pts = np.array([x.tolist() for i,x in enumerate(contours[0][:, 0, :]) if i % 2 == 0 ], dtype=np.int64) 
                else:
                    RV_myo_pts = []

                # Get intersection points between RV epi and LV epi
                if len(RV_epi_pts)>0 and len(RV_myo_pts)>0:

                    pairs = self.get_intersections(RV_epi_pts, RV_myo_pts, distance_cutoff = 10.0)

                    if len(pairs) > 0:
                        RV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_epi_pts) if i in np.unique(pairs[:,0])], 
                                              dtype=np.int64)
                        
                # Remove points located in the valve planes
                landmarks = self.landmarks_df[self.landmarks_df['Slice ID'] == int(slice_id)]
                landmarks = landmarks[landmarks['Time Frame'] == time]
                
                # pulmonary valve
                pv1 = landmarks['PV1'].item()
                pv2 = landmarks['PV2'].item()
                RV_epi_pts = remove_valve_points(RV_epi_pts, [pv1, pv2])

                if display:
                    try:
                        plt.figure(figsize=(12,12))
                        plt.imshow(image,cmap='gray')
                        plt.scatter(pv1[1], pv1[0])
                        plt.scatter(pv2[1], pv2[0])
                        plt.scatter(RV_epi_pts[:,1], RV_epi_pts[:,0], s=5, c='#0057E9')
                        plt.show()
                    except:
                        pass

                # transform to patient coordinates and write to GP file
                # only write the epi cardial points, as extracting the septum vs. freewall for the
                # endocardium is unreliable in the RVOT view
                point_lists = [RV_epi_pts] 
                labels = ['LAX_RV_EPICARDIAL']

                for i,points in enumerate(point_lists):
                    if len(points)>2:
                        pts = [inverse_coordinate_transformation(point, S, imgOrient, ps)
                                   for point in points.tolist()]

                        if time > 1:
                            # write to file
                            write_to_gp_file(self.output_folder + '/GP_ES.txt', pts, labels[i], slice_id, weight=1.0, phase=float(time))
                        else:
                            write_to_gp_file(self.output_folder + '/GP_ED.txt', pts, labels[i], slice_id, weight=1.0, phase=float(time))
            
            else:
                # 4CH and 3CH views are processed similarly
                # load necessary information for coordinate transformation
                _slice_info = self.slice_info_df[self.slice_info_df['Slice ID'] == int(slice_id)]
                S = _slice_info['ImagePositionPatient'].values[0]
                imgOrient = _slice_info['ImageOrientationPatient'].values[0]
                ps = _slice_info['Pixel Spacing'].values[0]

                # extract points
                LV_endo = (segmentation == 1).astype(np.uint8)
                LV_myo = (segmentation == 2).astype(np.uint8)
                LV_epi = (LV_endo | LV_myo).astype(np.uint8)
                RV_endo = (segmentation == 3).astype(np.uint8)
                RV_myo = (segmentation == 4).astype(np.uint8)
                RV_epi = (RV_endo | RV_myo).astype(np.uint8)

                # convert to contours
                contours, hierarchy = cv2.findContours(cv2.inRange(LV_endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                LV_endo_pts = np.array([x.tolist() for i,x in enumerate(contours[0][:, 0, :]) if i % 2 == 0 ], dtype=np.int64)  
                contours, hierarchy = cv2.findContours(cv2.inRange(LV_epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                LV_epi_pts = np.array([x.tolist() for i,x in enumerate(contours[0][:, 0, :]) if i % 2 == 0 ], dtype=np.int64) 
                contours, hierarchy = cv2.findContours(cv2.inRange(RV_endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                if len(contours) > 0:
                    RV_endo_pts = np.array([x.tolist() for i,x in enumerate(contours[0][:, 0, :]) if i % 2 == 0 ], dtype=np.int64) 
                else:
                    RV_endo_pts = []
                contours, hierarchy = cv2.findContours(cv2.inRange(RV_epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                if len(contours) > 0:
                    RV_epi_pts = np.array([x.tolist() for i,x in enumerate(contours[0][:, 0, :]) if i % 2 == 0 ], dtype=np.int64) 
                else:
                    RV_epi_pts = []

                # Get intersection points between RV endo and LV epi
                if len(RV_endo_pts)>0 and len(LV_epi_pts)>0:

                    pairs = self.get_intersections(RV_endo_pts, LV_epi_pts, distance_cutoff=4.5)

                    if len(pairs) > 0:
                        RV_septal_pts = RV_endo_pts[np.unique(pairs[:,0])] # deletes intersection from RV endo pts

                # Get intersection points between RV epi and LV epi
                if len(RV_epi_pts)>0 and len(LV_epi_pts)>0:

                    pairs = self.get_intersections(RV_epi_pts, LV_epi_pts, distance_cutoff=4.5)

                    if len(pairs) > 0:
                        intersection_points = LV_epi_pts[pairs[:,1],:] # intersection points on LV
                        intersection_on_LV_idx = np.unique(pairs[:,1])
                        LV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(LV_epi_pts) if i not in intersection_on_LV_idx], 
                                              dtype=np.int64)
                        RV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_epi_pts) if i not in np.unique(pairs[:,0])], 
                                              dtype=np.int64)

                # Remove points located in the valve planes
                landmarks = self.landmarks_df[self.landmarks_df['Slice ID'] == int(slice_id)]
                landmarks = landmarks[landmarks['Time Frame'] == time]
                
                if view == '4CH':
                    # mitral valve
                    mv1 = landmarks['MV1'].item()
                    mv2 = landmarks['MV2'].item()
                    LV_epi_pts = remove_valve_points(LV_epi_pts, [mv1, mv2])
                    LV_endo_pts = remove_valve_points(LV_endo_pts, [mv1, mv2])
                    
                    # tricuspid valve
                    tv1 = landmarks['TV1'].item()
                    tv2 = landmarks['TV2'].item()
                    RV_epi_pts = remove_valve_points(RV_epi_pts, [tv1, tv2])
                    RV_endo_pts = remove_valve_points(RV_endo_pts, [tv1, tv2])
                    RV_septal_pts = remove_valve_points(RV_septal_pts, [tv1, tv2])
                    
                if view == '3CH':
                    # mitral valve
                    mv1 = landmarks['MV1'].item()
                    mv2 = landmarks['MV2'].item()
                    LV_epi_pts = remove_valve_points(LV_epi_pts, [mv1, mv2])
                    LV_endo_pts = remove_valve_points(LV_endo_pts, [mv1, mv2])
                    
                    # aortic valve
                    av1 = landmarks['AV1'].item()
                    av2 = landmarks['AV2'].item()
                    LV_epi_pts = remove_valve_points(LV_epi_pts, [av1, av2])
                    LV_endo_pts = remove_valve_points(LV_endo_pts, [av1, av2])
                
                if display:
                    try:
                        plt.figure(figsize=(12,12))
                        plt.imshow(image,cmap='gray')
                        plt.scatter(RV_endo_pts[:,1], RV_endo_pts[:,0], s=5, c='#F2CA19')
                        plt.scatter(RV_septal_pts[:,1], RV_septal_pts[:,0], s=5, c='#E11845')
                        plt.scatter(RV_epi_pts[:,1], RV_epi_pts[:,0], s=5, c='#0057E9')
                        plt.scatter(LV_epi_pts[:,1], LV_epi_pts[:,0], s=5, c='#0057E9')
                        plt.scatter(LV_endo_pts[:,1], LV_endo_pts[:,0], s=5, c='#87E911')
                        plt.scatter(mv1[1], mv1[0])
                        plt.scatter(mv2[1], mv2[0])
                        plt.show()
                    except:
                        pass

                # transform to patient coordinates and write to GP file
                point_lists = [RV_epi_pts,
                               RV_endo_pts,
                              RV_septal_pts,
                              LV_epi_pts,
                              LV_endo_pts]
                labels = ['LAX_RV_EPICARDIAL',
                         'LAX_RV_FREEWALL',
                         'LAX_RV_SEPTUM',
                         'LAX_LV_EPICARDIAL',
                         'LAX_LV_ENDOCARDIAL']

                for i,points in enumerate(point_lists):
                    if len(points)>2:
                        pts = [inverse_coordinate_transformation(point, S, imgOrient, ps)
                                   for point in points.tolist()]

                        if time > 1:
                            # write to file
                            write_to_gp_file(self.output_folder + '/GP_ES.txt', pts, labels[i], slice_id, weight=1.0, phase=float(time))
                        else:
                            write_to_gp_file(self.output_folder + '/GP_ED.txt', pts, labels[i], slice_id, weight=1.0, phase=float(time))

    def process_landmarks(self):

        """ Extracts the landmark points and writes to guide point file """

        for i, row in self.landmarks_df.iterrows():
            slice_id = row['Slice ID']
            view = row['View']
            time = float(row['Time Frame'])
            
            # extract transform info
            slice_row = self.slice_info_df[self.slice_info_df['Slice ID'] == slice_id].copy()
            S = slice_row['ImagePositionPatient'].values[0]
            imgOrient = slice_row['ImageOrientationPatient'].values[0]
            ps = slice_row['Pixel Spacing'].values[0]
            
            if view == 'SA':
                rv1 = row['RV1']
                rv2 = row['RV2']

                # transform point
                p1 = inverse_coordinate_transformation(rv1, S, imgOrient, ps)
                p2 = inverse_coordinate_transformation(rv2, S, imgOrient, ps)

                if time > 1:
                    write_to_gp_file(self.output_folder + '/GP_ES.txt', [p1,p2], 'RV_INSERT', slice_id, weight=1.0, phase=time)
                else:
                    write_to_gp_file(self.output_folder + '/GP_ED.txt', [p1,p2], 'RV_INSERT', slice_id, weight=1.0, phase=time)
            
            if view == '4CH':
                mv1 = row['MV1']
                mv2 = row['MV2']
                tv1 = row['TV1']
                tv2 = row['TV2']
                lva = row['LVA']

                # transform point
                p1 = inverse_coordinate_transformation(mv1, S, imgOrient, ps)
                p2 = inverse_coordinate_transformation(mv2, S, imgOrient, ps)
                p3 = inverse_coordinate_transformation(tv1, S, imgOrient, ps)
                p4 = inverse_coordinate_transformation(tv2, S, imgOrient, ps)
                p5 = inverse_coordinate_transformation(lva, S, imgOrient, ps)
                
                if time > 1:
                    write_to_gp_file(self.output_folder + '/GP_ES.txt', [p1,p2], 'MITRAL_VALVE', slice_id, weight=1.0, phase=time)
                    write_to_gp_file(self.output_folder + '/GP_ES.txt', [p3,p4], 'TRICUSPID_VALVE', slice_id, weight=1.0, phase=time)
                    write_to_gp_file(self.output_folder + '/GP_ES.txt', [p5], 'APEX_POINT', slice_id, weight=1.0, phase=time)
                else:
                    write_to_gp_file(self.output_folder + '/GP_ED.txt', [p1,p2], 'MITRAL_VALVE', slice_id, weight=1.0, phase=time)
                    write_to_gp_file(self.output_folder + '/GP_ED.txt', [p3,p4], 'TRICUSPID_VALVE', slice_id, weight=1.0, phase=time)
                    write_to_gp_file(self.output_folder + '/GP_ED.txt', [p5], 'APEX_POINT', slice_id, weight=1.0, phase=time)
                    
            if view == '3CH':
                mv1 = row['MV1']
                mv2 = row['MV2']
                av1 = row['AV1']
                av2 = row['AV2']

                # transform point
                p1 = inverse_coordinate_transformation(mv1, S, imgOrient, ps)
                p2 = inverse_coordinate_transformation(mv2, S, imgOrient, ps)
                p3 = inverse_coordinate_transformation(av1, S, imgOrient, ps)
                p4 = inverse_coordinate_transformation(av2, S, imgOrient, ps)

                if time > 1:
                    write_to_gp_file(self.output_folder + '/GP_ES.txt', [p1,p2], 'MITRAL_VALVE', slice_id, weight=1.0, phase=time)
                    write_to_gp_file(self.output_folder + '/GP_ES.txt', [p3,p4], 'AORTA_VALVE', slice_id, weight=1.0, phase=time)
                else:
                    write_to_gp_file(self.output_folder + '/GP_ED.txt', [p1,p2], 'MITRAL_VALVE', slice_id, weight=1.0, phase=time)
                    write_to_gp_file(self.output_folder + '/GP_ED.txt', [p3,p4], 'AORTA_VALVE', slice_id, weight=1.0, phase=time)

            if view == 'RVOT':
                pv1 = row['PV1']
                pv2 = row['PV2']

                # transform point
                p1 = inverse_coordinate_transformation(pv1, S, imgOrient, ps)
                p2 = inverse_coordinate_transformation(pv2, S, imgOrient, ps)

                if time > 1:
                    write_to_gp_file(self.output_folder + '/GP_ES.txt', [p1,p2], 'PULMONARY_VALVE', slice_id, weight=1.0, phase=time)
                else:
                    write_to_gp_file(self.output_folder + '/GP_ED.txt', [p1,p2], 'PULMONARY_VALVE', slice_id, weight=1.0, phase=time)


    def extract_guidepoints(self, display=False):

        """ Main function that takes segmentations, extracts contours, and writes to GP file """

        if os.path.exists(self.output_folder + '/GP_ED.txt'):
            print('Guide point file already exists at {}'.format(self.output_folder + '/GP_ED.txt'))
            print('Removing file, and regenerating guide points..')
            try:
                os.remove(self.output_folder + '/GP_ED.txt')
            except:
                pass

        if os.path.exists(self.output_folder + '/GP_ES.txt'):
            print('Guide point file already exists at {}'.format(self.output_folder + '/GP_ES.txt'))
            print('Removing file, and regenerating guide points..')
            try:
                os.remove(self.output_folder + '/GP_ES.txt')
            except:
                pass

        self.get_segmentation_files()
        self.process_short_axis(display=display)
        self.process_long_axis(display=display)
        self.process_landmarks()