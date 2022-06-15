#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pydicom
from multiprocessing import Pool
import numpy as np
import pandas as pd
import tensorflow as tf


def clean_text(string):

    # clean and standardize text descriptions, which makes searching files easier

    forbidden_symbols = ["*", ".", ",", "\"", "\\", "/", "|", "[", "]", ":", ";", " "]
    for symbol in forbidden_symbols:
        string = string.replace(symbol, "_")  # replace everything with an underscore

    return string.lower()


def preprocess(img):

    # format image into tensor, standardized to 0-255

    img = tf.cast(img, tf.float32)
    img = tf.image.resize(tf.expand_dims(img, 2), (224, 224))
    img = tf.image.grayscale_to_rgb(img)

    # standardize
    img = img / np.max(img)
    img = img * 255.

    return img


class ViewSelection:

    # Class Instance that performs view prediction and selection for a raw MRI dump.

    # Class Variables
    class_labels = ['SA', '3CH', '4CH', '2CH RT', 'RVOT', 'OTHER', '2CH LT', 'LVOT']
    classes = sorted(class_labels, key=str)
    desired_series = ['3CH', '4CH', 'SA', 'LVOT', 'RVOT', '2CH RT', '2CH LT']
    model_path = '../models/'

    # init method
    def __init__(self, directory, dst, model_name,
                 csv_path,
                 create_csv,
                 use_multiprocessing,
                 save_files,
                 save_only_desired,
                 confidence_value):

            # Instance Variables
            self.directory = directory                          # (str) directory containing dicoms
            self.dst = dst                                      # (str) destination directory to save files to
            self.model_name = model_name                        # (str) the model used to make predictions
            self.csv_path = csv_path                            # (str) output csv path
            self.create_csv = create_csv                        # (boolean) whether to save a csv of predictions
            self.use_multiprocessing = use_multiprocessing      # (boolean) use multiprocessing to read header info
            self.save_files = save_files                        # (boolean) save dicoms to new folder (dst)
            self.save_only_desired = save_only_desired          # (boolean) save only desired views
            self.confidence_value = confidence_value            # (float) confidence value required for positive pred.
            self.model = None

    def load_tensorflow_model(self, model_path=model_path):

        # Loads the appropriate tensorflow model

        if self.model_name == 'ResNet50':
            self.model = tf.keras.models.load_model(os.path.join(model_path, 'ViewSelection/resnet50.hdf5'))
            # print(model.summary())

        elif self.model_name == 'VGG19':
            self.model = tf.keras.models.load_model(os.path.join(model_path, 'ViewSelection/vgg19.hdf5'))
            # print(model.summary())

        elif self.model_name == 'Xception':
            self.model = tf.keras.models.load_model(os.path.join(model_path, 'ViewSelection/xception.hdf5'))
            # print(model.summary())

        else:
            print('Unknown model specified in parameters!')

    def predict_view(self, img, classes=classes):

        # make prediction on a single image

        prediction = self.model.predict(tf.expand_dims(img, axis=0))
        prediction = tf.argmax(prediction, axis=-1)
        predicted_view = classes[int(prediction)]

        return predicted_view

    def batch_predict(self, batch, classes=classes):

        # make prediction on a batch of images

        prediction = self.model.predict(batch)
        prediction = tf.argmax(prediction, axis=-1)
        predicted_view = [classes[int(x)] for x in prediction]

        return predicted_view

    def get_dicom_header(self, dicom_loc):

        # read dicom file and return header information and image

        ds = pydicom.read_file(dicom_loc, force=True)

        # get patient, study, and series information
        patient_id = clean_text(ds.get("PatientID", "NA"))
        series_description = clean_text(ds.get("SeriesDescription", "NA"))

        # generate new, standardized file name
        modality = ds.get("Modality","NA")
        series_instance_uid = ds.get("SeriesInstanceUID","NA")
        series_number = ds.get('SeriesNumber', 'NA')
        instance_number = str(ds.get("InstanceNumber","0"))
        image_position_patient = str(ds.get("ImagePositionPatient", 'NA'))

        # load image data
        array = ds.pixel_array

        return patient_id, dicom_loc, modality, series_instance_uid, \
               series_number, instance_number, image_position_patient, array, series_description

    def complete_view_prediction(self):

        # Runs the complete view prediction over the dicom files in a directory

        unsorted_list = []
        for root, dirs, files in os.walk(self.directory):
            for file in files:
                if ".dcm" in file: # exclude non-dicoms, good for messy folders
                    unsorted_list.append(os.path.join(root, file))

        if self.use_multiprocessing:
            with Pool(os.cpu_count()) as p:
                output = p.map(self.get_dicom_header, [dicom_loc for dicom_loc in unsorted_list])

        else:
            output = []
            for dicom_loc in unsorted_list:
                output.append(self.get_dicom_header(dicom_loc))

        # generated pandas dataframe to store information from headers
        df = pd.DataFrame(sorted(output), columns=['Patient ID',
                                                     'Filename',
                                                     'Modality',
                                                     'Series ID',
                                                     'Series Number',
                                                     'Instance Number',
                                                     'Image Position Patient', 
                                                     'Img',
                                                     'Series Description'])

        output_series = []
        # make predictions and calculate confidence values
        for series in set(df['Series ID']):
            new = df[df['Series ID'] == series]

            dataset = tf.data.Dataset.from_tensor_slices([preprocess(x) for x in new['Img'].values])
            dataset = (dataset
                     .batch(16)
                     .prefetch(tf.data.experimental.AUTOTUNE))

            # record info for this series
            patient_id = new['Patient ID'].iloc[0]
            series_num = new['Series Number'].iloc[0]
            series_desc = new['Series Description'].iloc[0]
            frames = len(new)
            
            # get frames per slice
            frames_per_slice = int(np.mean(new['Image Position Patient'].value_counts()))

            # make predictions over images
            views = self.batch_predict(dataset)

            # find unique predictions and confidence for that series
            u, count = np.unique(views, return_counts=True)
            count_sort_ind = np.argsort(-count)
            prediction= u[count_sort_ind][0]
            conf = np.round(np.max(count) / np.sum(count), 2)

            output_series.append([patient_id.upper(), series, series_num, frames, frames_per_slice, series_desc, prediction, conf])

        output_series_df = pd.DataFrame(output_series, columns=['Patient ID',
                                                                'Series ID',
                                                                'Series Number',
                                                                'Frames',
                                                                'Frames Per Slice',
                                                                'Series Description',
                                                                'Predicted View',
                                                                'Confidence'])

        if self.create_csv:
            # print('Saving .csv file with series predictions and info')
            output_series_df.to_csv(self.csv_path, mode='w', index=False)
            # print('Done!')

        if self.save_files:
            # print('Saving dicom files to new folder...')
            for series in output_series_df['Series ID']:
                new = df[df['Series ID'] == series]
                series_df = output_series_df[output_series_df['Series ID'] == series]
                predicted_view = series_df['Predicted View'].values[0]
                patient_id = series_df['Patient ID'].values[0].upper()

                if self.save_only_desired:
                    if predicted_view in self.desired_series and \
                                    series_df['Confidence'].values > self.confidence_value:
                        for i, row in new.iterrows():
                            file_name = row['Modality'] + '.' + \
                                            row['Series ID'] + '.' + row['Instance Number'] + '.dcm'
                            dicom = pydicom.dcmread(row['Filename'])

                            # save files to a 2-tier nested folder structure
                            if not os.path.exists(os.path.join(self.dst, patient_id)):
                                os.makedirs(os.path.join(self.dst, patient_id))

                            if not os.path.exists(os.path.join(self.dst, patient_id, predicted_view)):
                                os.makedirs(os.path.join(self.dst, patient_id, predicted_view))

                            dicom.save_as(os.path.join(self.dst, patient_id, predicted_view, file_name))
                    else:
                        pass
                else:
                    for i, row in new.iterrows():
                        file_name = row['Modality'] + '.' + \
                                    row['Series ID'] + '.' + row['Instance Number'] + '.dcm'
                        patient_id = row['Patient ID'].upper()
                        dicom = pydicom.dcmread(row['Filename'])

                        # save files to a 2-tier nested folder structure
                        if not os.path.exists(os.path.join(self.dst, patient_id)):
                            os.makedirs(os.path.join(self.dst, patient_id))

                        if not os.path.exists(os.path.join(self.dst, patient_id, predicted_view)):
                            os.makedirs(os.path.join(self.dst, patient_id, predicted_view))

                        dicom.save_as(os.path.join(self.dst, patient_id, predicted_view, file_name))
