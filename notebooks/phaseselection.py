#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

    def __init__(self, dicom_list, series_list):

        self.dicom_list = dicom_list
        self.series_list = series_list
        self.series_headers = None
        self.loaded_dcm_list = None
        self.model_path = "../models/PhaseSelection/resnet50_lstm.hdf5"
        self.model = None
        self.volume = None

    def load_tensorflow_model(self):

        # initializes a tensorflow model

        self.model = tf.keras.models.load_model(self.model_path)

    def get_series_headers(self):

        # load dicom files for each series in a list of series IDs.

        self.series_headers = []

        print("Loading dicom header information...")
        for dicom_loc in tqdm(self.dicom_list):
            # read dicom file and return header information and image
            ds = pydicom.read_file(dicom_loc, force=True)
            series_instance_uid = ds.get("SeriesInstanceUID", "NA")

            if series_instance_uid in self.series_list:
                # get patient, study, and series information
                patient_id = clean_text(ds.get("PatientID", "NA"))
                series_instance_uid = ds.get("SeriesInstanceUID", "NA")
                instance_number = str(ds.get("InstanceNumber", "0"))
                slice_location = str(ds.get("SliceLocation", "NA"))

                # load image data
                array = ds.pixel_array

                self.series_headers.append(
                    [
                        patient_id,
                        dicom_loc,
                        series_instance_uid,
                        instance_number,
                        slice_location,
                        array,
                    ]
                )

    def volume_from_instance_idx(self):

        # Loads a 3D dicom image volume with the phase slice
        # (or instances if phases are unavailable) defining each volume

        # save image dimensions, window, and tags from the first image in the list
        imshape = (224, 224)
        window_center = self.loaded_dcm_list[0][0x0028, 0x1050].value
        window_width = self.loaded_dcm_list[0][0x0028, 0x1051].value

        # store slice locations in list
        slice_locations = [dcm[0x0020, 0x1041].value for dcm in self.loaded_dcm_list]

        # make map to convert slice locations into consecutive integers
        map_dict = {}
        for i, loc in enumerate(sorted(list(set(slice_locations)))):
            map_dict[loc] = i

        # create a volume of the appropriate size
        self.volume = np.zeros(
            (int(len(set(slice_locations))), 30, imshape[0], imshape[1], 1),
            dtype=np.uint16,
        )

        reversed_map = False

        # iterate through dicoms and assign pixel_array to the appropriate location in the slice
        for dcm in self.loaded_dcm_list:
            slice_loc = dcm[0x0020, 0x1041].value
            slice_idx = map_dict[slice_loc]
            phase_idx = int(dcm.InstanceNumber - 30 * slice_idx) - 1

            if phase_idx > 30 or phase_idx < 0:
                # will need to reverse the order of slice locations
                map_dict = {}
                for i, loc in enumerate(reversed(sorted(list(set(slice_locations))))):
                    map_dict[loc] = i

                reversed_map = True

                # reiterate through dcms
                for dcm in self.loaded_dcm_list:
                    slice_loc = dcm[0x0020, 0x1041].value
                    slice_idx = map_dict[slice_loc]
                    phase_idx = int(dcm.InstanceNumber - 30 * slice_idx) - 1

                    img = windowing(
                        dcm.pixel_array, window_center, window_width
                    ).astype(np.uint16)
                    resized = cv2.resize(
                        img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC
                    )

                    # insert into volume
                    self.volume[slice_idx, phase_idx, :, :, 0] = np.copy(resized)

            elif reversed_map == False:

                img = windowing(dcm.pixel_array, window_center, window_width).astype(
                    np.uint16
                )
                resized = cv2.resize(
                    img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC
                )

                # insert into volume
                self.volume[slice_idx, phase_idx, :, :, 0] = np.copy(resized)

            else:
                break

    def generator(self, counter=0, window_size=30):

        # Data generator for 2D   time images of shape (None, batch_sz, 30, 224, 224, 1)"

        keys = [x for x in range(self.volume.shape[0])]

        while True:
            input_images = np.zeros((5, window_size, 224, 224, 3))

            for i in range(5):
                # load images and concatenate along batch axis
                imgs = np.array(self.volume[keys[counter], ...]).astype(np.float32)
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

        # load series header
        self.get_series_headers()

        # generate intermediate dataframe to store dicom information
        temp_df = pd.DataFrame(
            self.series_headers,
            columns=[
                "Patient ID",
                "Filepath",
                "Series ID",
                "Instance ID",
                "Slice Location",
                "Array",
            ],
        )

        prediction_dictionary = {}
        for series in temp_df["Series ID"].unique():
            series_df = temp_df.loc[temp_df["Series ID"] == series]

            files = series_df["Filepath"]
            self.loaded_dcm_list = []
            # iterate through files in current list
            for file in files:
                # check to ensure file is a valid dcm file
                try:
                    dcm = pydicom.dcmread(file)
                    self.loaded_dcm_list.append(dcm)
                except IOError:
                    pass

            # generate volume from individual dicoms
            self.volume_from_instance_idx()

            if self.volume is not None:
                predicted_es = []

                slices = self.volume.shape[0]
                if slices > 7:  # exclude most apical and basal slices if possible
                    slice_max = self.volume.shape[0] - 2
                    slice_min = 2
                else:
                    slice_max = self.volume.shape[0]
                    slice_min = 0

                print(
                    "\nMaking ES phase predictions for each slice in the short-axis stack..."
                )
                for j in tqdm(range(slice_min, slice_max)):
                    predictions = np.zeros((5, 30))
                    images = next(
                        self.generator(counter=j)
                    )  # for each volume, generate the corresponding input

                    for i, img in enumerate(images):
                        predictions_raw = self.model.predict_on_batch(
                            tf.expand_dims(img, 0)
                        )
                        predictions[i, :] = np.roll(
                            tf.squeeze(predictions_raw), -i * 2, 0
                        )

                    predicted_es.append(np.argmax(np.mean(predictions, axis=0)))

            # append to prediction dictionary
            prediction_dictionary[series] = np.median(predicted_es)

        return prediction_dictionary
