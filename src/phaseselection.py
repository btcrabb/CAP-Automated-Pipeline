#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import pydicom
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

from tensorflow.keras.applications import resnet50
from tqdm import tqdm

pd.options.mode.chained_assignment = None  # default='warn'


def clean_text(string):

    # Clean and standardize text descriptions, which makes searching files easier

    forbidden_symbols = ["*", ".", ",", '"', "\\", "/", "|", "[", "]", ":", ";", " "]
    for symbol in forbidden_symbols:
        string = string.replace(symbol, "_")  # replace everything with an underscore

    return string.lower()


def windowing(image, window_center, window_width):

    # Clip an array to the appropriate range given a window width and level

    # calculate min and max pixel values
    min_value = window_center - window_width / 2
    max_value = window_center + window_width / 2

    # clip values to appropriate window
    return np.clip(image, min_value, max_value)


class PhaseSelection:

    # Class that selects the ES phase from a series of short-axis cardiac images

    def __init__(self, 
                 dst, 
                 slice_info, 
                 view):

        # Instance Variables
        self.dst = dst  # (str) destination directory to save files to
        self.slice_info = slice_info    # (DataFrame) Pandas dataframe of slice info file
        self.view = view    # (str) the cardiac view
        self.model_path = "../models/PhaseSelection/resnet50_lstm.hdf5"
        self.model = None
        self.volume = None
        self.num_phases = 30

    def load_tensorflow_model(self):

        # initializes a tensorflow model

        self.model = tf.keras.models.load_model(self.model_path)

    def generate_mapping_dictionary(self):

        # Generates a mapping dictionary to go from slice ID (in slice info file) to ImagePositionPatient

        self.mapping_dict = {}
        for i, row in self.slice_info.iterrows():
            slice_id = row['Slice ID']
            location = row['ImagePositionPatient']
            self.mapping_dict[str(location)] = slice_id

    def preprocess_image(self, img, size, padColor=0, flip=False):
        
        # Preprocesses image - enables flipping (to change orientation of image), pads to square and resizes, and converts to uint8
        
        # converts image to uint8
        img = img / np.max(img)
        img = img*255
        img = img.astype(np.uint8)
        
        # pads and image to square and resizes
        h, w = img.shape[:2]
        sh, sw = size

        # interpolation method
        if h > sh or w > sw: # shrinking image
            interp = cv2.INTER_AREA
        else: # stretching image
            interp = cv2.INTER_CUBIC

        # aspect ratio of image
        aspect = w/h 

        # compute scaling and pad sizing
        if aspect > 1: # horizontal image
            new_w = sw
            new_h = np.round(new_w/aspect).astype(int)
            pad_vert = (sh-new_h)/2
            pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
            pad_left, pad_right = 0, 0
        elif aspect < 1: # vertical image
            new_h = sh
            new_w = np.round(new_h*aspect).astype(int)
            pad_horz = (sw-new_w)/2
            pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
            pad_top, pad_bot = 0, 0
        else: # square image
            new_h, new_w = sh, sw
            pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

        # set pad color
        if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
            padColor = [padColor]*3

        # scale and pad
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

        return scaled_img

    def generate_complete_volume(self):

        # Generates a volume to store 3D + time data for each cardiac view (4CH, 3CH, RVOT, SA)

        # build volume to store data
        slices = len(self.mapping_dict.keys()) + 2
        self.volume = np.zeros((slices, self.num_phases, 224, 224, 1), dtype=np.uint16)

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
            instances = [int(dcm.InstanceNumber) for dcm in loaded_dcm_dict[key]]
            min_instance = np.min(instances)

            try:
                for dcm in loaded_dcm_dict[key]:
                    instance_number = int(dcm.InstanceNumber)
                    phase_number = instance_number - min_instance

                    # add to volume
                    self.volume[slice_id, phase_number, :, :, 0] = self.preprocess_image(dcm.pixel_array, (224, 224))
            except:
                print('Unable to copy pixel array to volume. Likely caused by an incorrect phase number, check that your list of dicoms is correct\n')
                print('Information: ')
                print('Phase: ', phase_number)
                print('Slice number: ', slice_id)
                
                print('\nPossible sources of error: \nDuplicate series listed in slice info file (the dicoms are duplicated and stored in two places')
                print('The number of phases per slice is incorrect, or inconsistent between views')

    def generator(self, counter=0, window_size=30):

        # Data generator for 2D   time images of shape (None, batch_sz, 30, 224, 224, 1)"

        keys = [x for x in range(self.volume.shape[0])]

        while True:
            input_images = np.zeros((5, window_size, 224, 224, 3))

            for i in range(5):
                # load images and concatenate along batch axis
                imgs = np.array(self.volume[keys[counter], :30, ...]).astype(np.float32)
                imgs = np.roll(imgs, i * 2, 0)

                frame2 = np.roll(
                    imgs, -1, 0
                )  # get next 2 frames (t 1, t 2) to append to the original image
                frame3 = np.roll(imgs, -2, 0)
                imgs = tf.squeeze(np.stack((imgs, frame2, frame3), axis=3))

                imgs = np.expand_dims(imgs, 0)
                input_images[i] = np.concatenate(imgs, axis=0)

                # normalize images
                input_images[i] = input_images[i] - np.min(input_images[i])

                max_value = np.max(input_images[i])
                if max_value > 0:
                    input_images[i] = (input_images[i] / max_value) * 255.0
                else:
                    pass

                input_images[i] = resnet50.preprocess_input(
                    input_images[i]
                )

            yield (input_images.astype(np.float32))
            counter = 1

    def predict_phase(self):

        # predicts the ES phase for a list of dicoms

        # convert slice index to slice locations
        view_df = self.slice_info[self.slice_info['View'] == self.view]
        sax_slice_ids = view_df['Slice ID']

        if self.volume is not None:
            predicted_es = []

            # exclude most apical and basal slices if possible
            slices = len(sax_slice_ids)
            if slices > 7:  
                slice_max = np.max(sax_slice_ids) - 2
                slice_min = np.min(sax_slice_ids) + 2
            else:
                slice_max = np.max(sax_slice_ids)
                slice_min = np.min(sax_slice_ids)

                print(
                    "\nMaking ES phase predictions for each slice in the short-axis stack..."
                )

            # iterate through selected slices
            for j in tqdm(range(slice_min, slice_max)):
                predictions = np.zeros((5, 30))
                images = next(
                    self.generator(counter=j)
                )  # for each volume, generate the corresponding input

                # predict on batch for each generated input
                for i, img in enumerate(images):
                    predictions_raw = self.model.predict_on_batch(
                        tf.expand_dims(img, 0)
                    )
                    predictions[i, :] = np.roll(
                        tf.squeeze(predictions_raw), -i * 2, 0
                    )

                predicted_es.append(np.ceil(np.argmax(np.mean(predictions, axis=0))))

            # average predictions to find es phase
            es_phase = int(np.ceil(np.median(predicted_es)))

        return es_phase
