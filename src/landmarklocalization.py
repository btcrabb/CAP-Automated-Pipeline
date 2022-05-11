#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


def loss(y_true, y_pred):

    # placeholder to load tensorflow model (not necessary to define as training won't be used)

    return None


class LandmarkLocalization:

    # Class that performs landmark localization in 3CH, 4CH, RVOT, and SA cardiac views

    # Class Variables
    class_labels = ['SA', '3CH', '4CH', '2CH RT', 'RVOT', 'OTHER', '2CH LT', 'LVOT']
    classes = sorted(class_labels, key=str)
    desired_series = ['3CH', '4CH', 'SA', 'LVOT', 'RVOT']
    model_path = '../models/'

    # init method
    def __init__(self,
                 dst,
                 slice_info,
                 view):

        # Instance Variables
        self.dst = dst  # (str) destination directory to save files to
        self.slice_info = slice_info    # (DataFrame) Pandas dataframe of slice info file
        self.view = view    # (str) the cardiac view
        self.model = None
        self.mapping_dict = None
        self.volume = None
        self.heatmap_predictions = None
        self.point_predictions = None
        self.output = None

    def load_tensorflow_model(self, model_path=model_path):

        # Loads the appropriate tensorflow model

        if self.view == 'SA':
            self.model = tf.keras.models.load_model(os.path.join(model_path, 'Landmarks/SA.hdf5'),
                                                                 custom_objects={"custom_dice":loss})

        elif self.view == '4CH':
            self.model = tf.keras.models.load_model(os.path.join(model_path, 'Landmarks/4CH.hdf5'),
                                                                 custom_objects={"custom_dice":loss})

        elif self.view == '3CH':
            self.model = tf.keras.models.load_model(os.path.join(model_path, 'Landmarks/3CH.hdf5'),
                                                                 custom_objects={"custom_dice":loss})

        elif self.view == 'RVOT':
            self.model = tf.keras.models.load_model(os.path.join(model_path, 'Landmarks/RVOT.hdf5'),
                                                                 custom_objects={"custom_dice":loss})

        else:
            print('Unknown model specified in parameters!')

    def generate_mapping_dictionary(self):

        # Generates a mapping dictionary to go from slice ID (in slice info file) to ImagePositionPatient

        self.mapping_dict = {}
        for i, row in self.slice_info.iterrows():
            slice_id = row['Slice ID']
            location = row['ImagePositionPatient']
            self.mapping_dict[str(location)] = slice_id

    def generate_complete_volume(self):

        # Generates a volume to store 3D + time data for each cardiac view (4CH, 3CH, RVOT, SA)

        # build volume to store data
        slices = len(self.mapping_dict.keys())
        self.volume = np.zeros((slices, 30, 256, 256, 1), dtype=np.uint16)

        # find associated dicoms
        loaded_dcm_dict = {}
        for (root, subdir, files) in os.walk(os.path.join(self.dst)):
            for file in files:
                if '.dcm' in file:
                    dcm = pydicom.dcmread(os.path.join(root, file), force=True)
                    location = str(dcm.ImagePositionPatient)

                    if location in self.mapping_dict.keys():
                        if location not in loaded_dcm_dict.keys():
                            loaded_dcm_dict[location] = [dcm]
                        else:
                            loaded_dcm_dict[location].append(dcm)

        # iterate through each slice location, ordering images and adding to volume
        for key in loaded_dcm_dict.keys():
            slice_id = int(self.mapping_dict[key])
            instances = [dcm.InstanceNumber for dcm in loaded_dcm_dict[key]]
            min_instance = np.min(instances)

            for dcm in loaded_dcm_dict[key]:
                instance_number = int(dcm.InstanceNumber)
                phase_number = instance_number - min_instance

                # add to volume
                self.volume[slice_id, phase_number, :, :, 0] = dcm.pixel_array

    def predict_landmarks(self, phase=0):

        # Predicts the landmarks for the specified view

        # convert slice index to slice locations
        view_df = self.slice_info[self.slice_info['View'] == self.view]
        slice_locations = view_df['Slice Location'].unique()

        sorted_slice_locations = sorted(slice_locations)
        view_mapping_dict = {}
        for i, loc in enumerate(sorted_slice_locations):
            view_mapping_dict[loc] = float(i)

        self.output = []
        for i, row in self.slice_info.iterrows():

            if row['View'] == self.view:
                slice_id = row['Slice ID']
                mri_slice = self.volume[slice_id, ...]
                # print('Slice ID - {} for location {}'.format(slice_id, row['Slice Location']))

                # perform prediction at one phase
                # create rolled input
                f0 = np.roll(mri_slice, -2, axis=0)[phase, :, :, 0]
                f1 = np.roll(mri_slice, -1, axis=0)[phase, :, :, 0]
                f2 = np.roll(mri_slice, 0, axis=0)[phase, :, :, 0]
                f3 = np.roll(mri_slice, 1, axis=0)[phase, :, :, 0]
                f4 = np.roll(mri_slice, 2, axis=0)[phase, :, :, 0]

                inputs = np.stack([f0, f1, f2, f3, f4], axis=-1)
                img = inputs / np.amax(inputs)
                img = tf.expand_dims(tf.convert_to_tensor(img, dtype=tf.float32), 0)
                self.heatmap_predictions = self.model.predict(img)

                # mask borders of predictions
                self.heatmap_predictions['outputs'][:, :5, :5, :] = 0
                self.heatmap_predictions['outputs'][:, -5:, -5:, :] = 0

                # Extract point from heatmap and append each prediction to output df
                self.point_predictions = [slice_id, self.view, phase]
                for i in range(self.heatmap_predictions['outputs'].shape[-1]):

                    landmark = np.where(self.heatmap_predictions['outputs'][0, :, :, i] ==
                              np.max(self.heatmap_predictions['outputs'][0, :, :, i]))
                    self.point_predictions.append(landmark)

                self.output.append(self.point_predictions)




