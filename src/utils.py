#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import tensorflow as tf
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil

def preprocess(img):

    """ format image as float and resize """

    img = tf.cast(img, tf.float32)
    img = tf.image.resize(tf.expand_dims(img, 2), (224, 224))
    img = tf.image.grayscale_to_rgb(img)
    
    # standardize
    img = img / np.max(img)

    return img

def plot_view_classifications(dst, patient):


	""" Displays current image classification assignments in the output directory """

	plt.figure(figsize=(12,32))

	idx = 1
	for directory in os.listdir(os.path.join(dst,patient)):
	    
	    if os.path.isdir(os.path.join(dst, patient, directory)):
	        
	        # load first dicom
	        files = os.listdir(os.path.join(dst, patient, directory))
	        
	        series = []
	        for file in files:
	            dcm = pydicom.dcmread(os.path.join(dst, patient, directory, file))
	            if dcm.SeriesNumber not in series:
	                series.append(dcm.SeriesNumber)
	                series_num = dcm.SeriesNumber
	                image = preprocess(dcm.pixel_array)

	                ax = plt.subplot(8,4,idx)
	                ax.imshow(image, cmap='gray')
	                ax.set_xticks([])
	                ax.set_yticks([])
	                ax.set_title('Series Number {} - {}'.format(series_num, directory))

	                # increase index    
	                idx += 1
	        
	plt.show()

def manual_view_classification(labels_path, src, dst, patient):

	""" Uses manual labels saved in an excel file to select cardiac views """

	# standardize naming convention
	label_map = {
	    'SAX': 'SA',
	    '4CH': '4CH',
	    '2CH': '2CH LT',
	    '3CH': '3CH',
	    'RVOT': 'RVOT',
	    'RVT': '2CH RT',
	}

	# load excel file
	manual_labels = pd.read_excel(labels_path, engine='openpyxl')
	print('Manual labels loaded.')

	# make output directory if necessary
	if not os.path.isdir(os.path.join(dst, patient)):
	    os.mkdir(os.path.join(dst, patient))
	    
	# iterate through dicom files in src directory
	for (root, subdir, files) in os.walk(src):
	    
	    print('Found {} source dicoms'.format(len(files)))
	    print('Moving selected files...')
	    for file in files: 
	        
	        if '.dcm' in file:
	            dcm = pydicom.dcmread(os.path.join(src, file))
	            study_id = dcm.InstanceCreationDate # study ID is = to the instance creation date for venus cases
	            label = manual_labels[manual_labels['StudyID'] == int(study_id)]
	            
	            for col in manual_labels.columns:
	                if label[col].item() == float(dcm.SeriesNumber):
	                    output_dir = label_map[col]
	                    
	                    # make directory if necessary
	                    if not os.path.isdir(os.path.join(dst, patient, output_dir)):
	                        os.mkdir(os.path.join(dst, patient, output_dir))
	                        
	                    # move file
	                    shutil.move(os.path.join(src, file), os.path.join(dst, patient, output_dir, file))

	print('Files moved! Building dataframe...')
	
	# generate dataframe to store information
	out = []
	for (root, subdir, files) in os.walk(os.path.join(dst, patient)):
	    for dir in subdir:
	        if os.path.isdir(os.path.join(root, dir)):
	            files = os.listdir(os.path.join(root, dir))
	            dcm = pydicom.dcmread(os.path.join(root, dir, files[0]))
	            
	            series_id = dcm.get("SeriesInstanceUID", "na")
	            series_num = dcm.get("SeriesNumber", "na")
	            series_desc = dcm.get("SeriesDescription", "na")
	            predicted_view = dir
	            confidence = 1.0
	            frames = len(files)
	            frames_per_slice = dcm.get("CardiacNumberofImages", "na")
	            out.append([patient, series_id, series_num, frames, 30, series_desc, predicted_view, confidence])
	            
	df = pd.DataFrame(out, columns = ['Patient ID', 'Series ID',
	                                  'Series Number', 'Frames', 
	                                  'Frames Per Slice', 'Series Description', 
	                                  'Predicted View', 'Confidence'])
	return df


