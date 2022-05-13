from logging import exception
import os
import cv2
import shutil
from glob import glob
from numpy.core.defchararray import index
from pandas.core.arrays.sparse import dtype
from tqdm import tqdm
from natsort import natsorted

import pickle
import numpy as np
import pandas as pd

import nibabel as nib
from nibabel.nifti1 import unit_codes, xform_codes, data_type_codes

import pydicom
from pydicom.dicomio import read_file

import matplotlib.pyplot as plt

import parse_cvi42_xml

from scipy.io import loadmat

from nibabel.nifti1 import unit_codes, xform_codes, data_type_codes

import re


def save_nifti(affine, volume, hdr2, out_dir):
    imgobj = nib.nifti1.Nifti1Image(volume, affine)

    # header info: http://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h
    hdr = {
        'pixdim': hdr2['pixdim'],
        'toffset': hdr2['toffset'],
        'slice_duration': hdr2['slice_duration'],
        'xyzt_units': unit_codes['mm'] | unit_codes['msec'],
        'qform_code': xform_codes['aligned'],
        'sform_code': xform_codes['scanner'],
        'datatype': data_type_codes.code[np.float32]
    }
    for k, v in hdr.items():
        if k in imgobj.header:
            imgobj.header[k] = v

    nib.save(imgobj, out_dir)


def volume4D_to_frames3D(save_dir, source_image_sax, study_name, seq_name, format_str):
    # Export images
    img = nib.load(source_image_sax)
    img_fdata = img.get_fdata()
    _, _, Z, T = img.shape

    img_hdr = img.header
    img_hdr['dim'][4] = 1
    img_hdr['pixdim'][4] = 0

    # if vol_type == '3D':  # 3D volumes
    for fr in range(T):
        filename_img_fr = os.path.join(save_dir, '{}_{}_fr_{:02d}{}'.format(study_name, seq_name, fr, format_str))
        save_nifti(img.affine, img_fdata[:, :, :, fr], img_hdr, filename_img_fr)


def get_temporal_sequences(sequences_dir, trigger_time_min):
    for sequence_dir in sequences_dir[:]:
        dicom = pydicom.read_files(sequence_dir)  # read metadata from DICOM file
        if hasattr(dicom, 'TriggerTime'):
            trigger_time = float(dicom.TriggerTime)  # get trigger time for DICOM
        else:
            raise exception('[INFO] No TriggerTime found for current DICOM.')
        if trigger_time <= trigger_time_min:
            sequences_dir.remove(sequence_dir)

    return sequences_dir


def check_reacquired_stack(sequences_dir, ):
    dcm_per_seq = []
    for sequence_dir in sequences_dir:
        dcm_count = pydicom.read_file(sequence_dir).pixel_array.shape[0]
        dcm_per_seq.append(dcm_count)

    repeated_stack = False
    if (np.array(
            dcm_per_seq) > 150).all():  # 15 frames and 10 slices = 150 TODO: save length of dcm_per_seq for each sequence
        repeated_stack = True

    return repeated_stack


def convert_dicom_to_nifti(dicom_dirs, dest_image_dir=None, single_slice=False):
    X, Y, dx, dy, dz, z_locs, img_pos, axis_x, axis_y, tt, inst_nb, series_nb = [], [], [], [], [], [], [], [], [], [], [], []
    data = []
    SOPInstanceUID = []
    folder_belong_dcm = []
    for source_dicom_dir in dicom_dirs:
        folder_belong_dcm.append(source_dicom_dir.split('/')[-2])  # get folder (sequence) corresponding to each DICOM
        dicom_data = read_file(source_dicom_dir)
        series_nb.append(int(dicom_data.SeriesNumber))
        inst_nb.append(dicom_data.InstanceNumber)
        X.append(dicom_data.Columns)
        Y.append(dicom_data.Rows)
        dx.append(float(dicom_data.PixelSpacing[0]))
        dy.append(float(dicom_data.PixelSpacing[1]))

        # Determine the z spacing
        if hasattr(dicom_data, 'SliceThickness'):
            dz.append(float(dicom_data.SliceThickness))
        elif hasattr(dicom_data, 'SpacingBetweenSlices'):
            dz.append(float(dicom_data.SpacingBetweenSlices))
        else:
            raise Exception('Cannot find attribute SliceThickness or SpacingBetweenSlices.')
        # if hasattr(dicom_data, 'SpacingBetweenSlices'):
        #     dz.append(float(dicom_data.SpacingBetweenSlices))
        # elif hasattr(dicom_data, 'SliceThickness'):
        #     dz.append(float(dicom_data.SliceThickness))
        # else:
        #     raise Exception('Cannot find attribute SliceThickness or SpacingBetweenSlices.')

        if hasattr(dicom_data, 'SOPInstanceUID'):
            SOPInstanceUID.append(dicom_data.SOPInstanceUID)
        else:
            raise Exception('Cannot find attribute SOPInstanceUID.')

        # Image position
        z_locs.append(float(dicom_data.SliceLocation))  # SliceLocation: z-location
        pos_ul = np.array([float(x) for x in
                           dicom_data.ImagePositionPatient])  # ImagePositionPatient: xyz coordinates of the center of the upper-left voxel for current slice
        pos_ul[:2] = -pos_ul[:2]
        img_pos.append(pos_ul)

        if hasattr(dicom_data, 'RescaleSlope'):
            rslope = float(dicom_data.RescaleSlope)
        else:
            rslope = 1
        if hasattr(dicom_data, 'RescaleIntercept'):
            rinter = float(dicom_data.RescaleIntercept)
        else:
            rinter = 0

        # Image orientation
        ax = np.array([float(x) for x in dicom_data.ImageOrientationPatient[:3]])
        ay = np.array([float(x) for x in dicom_data.ImageOrientationPatient[3:]])
        ax[:2] = -ax[:2]
        ay[:2] = -ay[:2]
        axis_x.append(ax)
        axis_y.append(ay)
        tt.append(int(dicom_data.TriggerTime))
        try:
            validPixelArray = dicom_data.pixel_array is not None
        except:
            raise Exception('Cannot find a valid pixel_array.')

        pixelarray = dicom_data.pixel_array * rslope + rinter
        data.append(pixelarray)

    ##########################################
    # CREATE NIFTI VOLUME
    ##########################################
    X = np.array(X)
    Y = np.array(Y)
    dx = np.array(dx)
    dy = np.array(dy)
    dz = np.array(dz)
    series_nb = np.array(series_nb)
    axis_x = np.array(axis_x)
    axis_y = np.array(axis_y)
    inst_nb = np.array(inst_nb).astype(int)
    data = np.array(data)
    SOPInstanceUID = np.array(SOPInstanceUID)
    z_locs = np.array(z_locs)
    img_pos = np.array(img_pos)
    tt = np.array(tt)
    folder_belong_dcm = np.array(folder_belong_dcm)

    # Generate Tags
    if len(np.unique(X)) != 1:
        raise Exception('Size of X changes over dicoms!')
    else:
        X = np.unique(X)[0]

    if len(np.unique(Y)) != 1:
        raise Exception('Size of Y changes over dicoms!')
    else:
        Y = np.unique(Y)[0]

    if len(np.unique(dx)) != 1:
        raise Exception('Size of dx changes over dicoms!')
    else:
        dx = np.unique(dx)[0]

    if len(np.unique(dy)) != 1:
        raise Exception('Size of dy changes over dicoms!')
    else:
        dy = np.unique(dy)[0]

    if len(np.unique(dz)) != 1:
        raise Exception('Size of dz changes over dicoms!')
    else:
        dz = np.unique(dz)[0]

    axis_x = np.squeeze(np.unique(axis_x, axis=0))
    axis_y = np.squeeze(np.unique(axis_y, axis=0))

    if len(axis_x) == 3 and len(axis_y) == 3:
        data = np.transpose(data)
        _, idx = np.unique(z_locs, return_index=True)
        z_locs_unique = np.sort(z_locs[idx])
        Z = len(z_locs_unique)

        img_pos = np.unique(img_pos, axis=0)
        if Z > 1:
            if img_pos.shape[0] == Z:
                z_diff = np.diff(img_pos, axis=0)
                axis_z = np.mean(z_diff, axis=0)
                if (abs(z_diff - axis_z) > 10 * np.std(z_diff, axis=0)).any():
                    raise Exception('z-spacing between slices varies by more than 10 standard deviations!')
                axis_z = axis_z / np.linalg.norm(axis_z)  # divide by the norm to normalise 3-D vector
            else:
                raise Exception('z-locations do not correspond with slice locations!')
        else:
            axis_z = np.cross(axis_x, axis_y)

        # T = np.unique(inst_nb).shape[0]
        T = len(z_locs) // Z

        # Affine matrix which converts the voxel coordinate to world coordinate NOTE: currently sform affine
        affine = np.eye(4)
        affine[:3, 0] = axis_y
        affine[:3, 1] = axis_x
        affine[:3, 2] = -1 * axis_z
        affine[:3, 3] = img_pos[-1, :]

        volume = np.zeros((Y, X, Z, T), dtype=np.float32)
        dcm_files = np.zeros((Z, T), dtype=object)

        # Divide data by slices
        for z_loc_idx, z_loc in enumerate(z_locs_unique):
            a = np.where(z_locs == z_loc)[0]
            if len(inst_nb[a]) != T:
                unique_series_nb = np.unique(series_nb[a])
                unique_dir = np.unique(folder_belong_dcm[a])
                ind = -1
                for f, fn in enumerate(unique_dir):
                    # if fn in sequences_SA:
                    ind = f
                    print(fn)
                if ind != -1:
                    a = np.where([series_nb == unique_series_nb[ind]])[1]
                else:
                    a = np.where([series_nb == unique_series_nb.max()])[1]
            trigTime = tt[a]
            inNb = inst_nb[a]
            if (trigTime.argsort() == inNb.argsort()).all():
                auxData = np.squeeze(data[:, :, a]) if len(a) > 1 else data[:, :, a]
                auxData = auxData[:, :, trigTime.argsort()]
                volume[:, :, Z - z_loc_idx - 1, :] = np.transpose(auxData, (1, 0, 2))
                auxData = np.squeeze(SOPInstanceUID[a]) if len(a) > 1 else SOPInstanceUID[a]
                auxData = auxData[trigTime.argsort()]
                dcm_files[Z - z_loc_idx - 1, :] = auxData
            else:
                raise Exception('Mismatch between trigger time and instance number.')

        if len(np.unique(tt)) > 1:
            dt = np.unique(np.diff(tt))
            dt = dt[dt > 0]
            dt = int(round(np.mean(dt)))
            toffset = 0.0
        elif len(np.unique(tt)) == 1 or T == 1:
            raise Exception('Only one trigger time found.')

        imgobj = nib.nifti1.Nifti1Image(volume, affine)  # Create images from numpy arrays

        # header info: http://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h
        hdr = {
            'pixdim': np.array([1.0, dx, dy, dz, dt, 1.0, 1.0, 1.0], np.float32),
            # NOTE: The srow_* vectors are in the NIFTI_1 header.  Note that no use is made of pixdim[] in this method.
            'toffset': toffset,
            'slice_duration': dt,
            'xyzt_units': unit_codes['mm'] | unit_codes['msec'],
            'qform_code': xform_codes['aligned'],
            # Code-Labels for qform codes https://nipy.org/nibabel/nifti_images.html
            'sform_code': xform_codes['scanner'],
            # Code-Labels for sform codes https://nipy.org/nibabel/nifti_images.html
            'datatype': data_type_codes.code[np.float32]
        }
        for k, v in hdr.items():
            if k in imgobj.header:
                imgobj.header[k] = v
        if dest_image_dir is not None:
            nib.save(imgobj, dest_image_dir)

    else:
        raise Exception('Problem converting dicoms: wrong size for axis_x and axis_y')

    if not single_slice:
        return [Y, X, Z, T, dcm_files, volume, affine, hdr]
    else:
        return [Y, X, 1, T, imgobj, volume, affine, hdr]


def save_contours(Y, X, Z, T, cvi42_dir, dcm_files, volume, affine, hdr, dest_nifti_seg):
    """extract contours from wsx files, it's for Venus cases only just for now."""
    up = 4
    label = np.zeros((Y, X, Z, T), dtype='int16')

    for t in range(T):
        for z in range(Z):
            contour_pickle = os.path.join(cvi42_dir, f'{dcm_files[z, t]}.pickle')
            # Check whether there is a corresponding cvi42 contour file for each DICOM in dcm_files
            if os.path.exists(contour_pickle):
                with open(contour_pickle, 'rb') as f:
                    contours = pickle.load(f)
                    # Labels NOTE: print(contours.keys())
                    lv_endo = 1
                    lv_epi = 2
                    rv_endo = 3
                    # la = 4
                    # ra = 5

                    # Fill the contours in order:
                    #   1. LV endocardium
                    #   2. LV epicardium
                    #   3. RV endocardium
                    #   4. Left Atrium
                    #   5. Right Atrium

                    ordered_contours = []

                    if 'saepicardialContour' in contours:
                        ordered_contours += [(contours['saepicardialContour'], lv_epi)]
                    if 'laepicardialContour' in contours:
                        ordered_contours += [(contours['laepicardialContour'], lv_epi)]
                    if 'saepicardialOpenContour' in contours:
                        ordered_contours += [(contours['saepicardialOpenContour'], lv_epi)]

                    if 'saendocardialContour' in contours:
                        ordered_contours += [(contours['saendocardialContour'], lv_endo)]
                    if 'laendocardialContour' in contours:
                        ordered_contours += [(contours['laendocardialContour'], lv_endo)]
                    if 'saendocardialOpenContour' in contours:
                        ordered_contours += [(contours['saendocardialOpenContour'], lv_endo)]

                    if 'sarvendocardialContour' in contours:
                        ordered_contours += [(contours['sarvendocardialContour'], rv_endo)]
                    if 'larvendocardialContour' in contours:
                        ordered_contours += [(contours['larvendocardialContour'], rv_endo)]

                    # if 'laraContour' in contours:
                    #     ordered_contours += [(contours['laraContour'], ra)]
                    # if 'lalaContour' in contours:
                    #     ordered_contours += [(contours['lalaContour'], la)]

                    # cv2.fillPoly requires the contour coordinates to be integers.
                    # However, the contour coordinates are floating point number since
                    # they are drawn on an upsampled image by 4 times.
                    # We multiply it by 4 to be an integer. Then we perform fillPoly on
                    # the upsampled image as cvi42 does. This leads to a consistent volume
                    # measurement as cvi2. If we perform fillPoly on the original image, the
                    # volumes are often over-estimated by 5~10%.
                    # We found that it also looks better to fill polygons it on the upsampled
                    # space and then downsample the label map than fill on the original image.
                    lab_up = np.zeros((Y * up, X * up))

                    for c, l in ordered_contours:
                        coord = np.round(c * up).astype(np.int)
                        cv2.fillPoly(lab_up, [coord], l)

                    if label[:, :, z, t].shape[0] == lab_up[::up, ::up].shape[0] and \
                            label[:, :, z, t].shape[1] == lab_up[::up, ::up].shape[1]:
                        label[:, :, z, t] = lab_up[::up, ::up]
                    elif label[:, :, z, t].shape[0] == lab_up[::up, ::up].shape[1] and \
                            label[:, :, z, t].shape[1] == lab_up[::up, ::up].shape[0]:
                        label[:, :, z, t] = lab_up[::up, ::up].T

    if label.shape[0] != volume.shape[0] or label.shape[1] != volume.shape[1]:
        raise Exception('Image and contorus have different size!')
    else:
        imgobj = nib.nifti1.Nifti1Image(label, affine)
        for k, v in hdr.items():
            if k in imgobj.header:
                imgobj.header[k] = v
        nib.save(imgobj, dest_nifti_seg)


def get_nifti(source_dir, dest_dir):
    patient_IDs = glob(f"{source_dir}/*")

    for patient_ID in tqdm(patient_IDs):
        # study folders with DICOMs
        study_IDs = glob(f"{patient_ID}/*/")
        # cvi42 files for the study
        cvi42_dirs = glob(f"{patient_ID}/*.cvi42wsx")

        for ID_idx, study_ID in enumerate(study_IDs):
            # destination folder for nifti files (volumes and contours)
            dest_ID_dir = os.path.join(dest_dir, *study_ID.split('/')[-4:])
            if not os.path.exists(dest_ID_dir):
                os.makedirs(dest_ID_dir)
            # create folder for temporary pickle files, which are manual contours
            pickles_dir = os.path.join(dest_ID_dir, 'cvi42_pickles')
            if not os.path.exists(pickles_dir):
                os.makedirs(pickles_dir)
            # convert wsx files to pickle files
            pickle_files = glob(f"{pickles_dir}/*.pickle")
            if len(pickle_files) == 0:
                # if cases in this study have been manually contoured
                # or if we have already converted those wsx files
                parse_cvi42_xml.parseFile(cvi42_dirs[ID_idx], pickles_dir)
            categories = os.listdir(study_ID)

            for category in categories:
                # get all series folders in a view folder
                folders = os.listdir(os.path.join(study_ID, category))

                for folder in folders:
                    # Bear in mind that each SAX series of the Venus includes all slices #
                    # swap the file name of dicoms with SOPInstanceUID
                    sequence_dirs = glob(os.path.join(study_ID, category, folder, '*.dcm'))
                    for sequence_dir in sequence_dirs:
                        dicom_file = pydicom.dcmread(sequence_dir)
                        if hasattr(dicom_file, 'SOPInstanceUID'):
                            SOPInstanceUID = dicom_file.SOPInstanceUID
                        else:
                            raise Exception('\rNo Series Instance UID in this dicom series.\n')
                        new_name = os.path.join('/', *sequence_dir.split('/')[:-1], SOPInstanceUID + '.dcm')
                        os.rename(sequence_dir, new_name)
                    sequence_dirs = natsorted(glob(os.path.join(study_ID, category, folder, '*.dcm')))
                    # convert dicom sequences to nifti files
                    dest_image_dir = os.path.join(dest_ID_dir, category)
                    if not os.path.exists(dest_image_dir):
                        os.makedirs(dest_image_dir)
                    dest_image_dir = os.path.join(dest_image_dir, f'{folder}_image.nii.gz')
                    data_nifti_image = convert_dicom_to_nifti(sequence_dirs, dest_image_dir)
                    Y, X, Z, T, dcm_files, volume, affine, hdr = data_nifti_image
                    # convert pickle masks to nifti files
                    dest_contour_nifti = os.path.join(dest_ID_dir, category)
                    if not os.path.exists(dest_contour_nifti):
                        os.makedirs(dest_contour_nifti)
                    dest_contour_nifti = os.path.join(dest_contour_nifti, f'{folder}_label.nii.gz')
                    save_contours(Y, X, Z, T, pickles_dir, dcm_files, volume, affine, hdr, dest_contour_nifti)
            shutil.rmtree(pickles_dir)


def save_contours_CAP(Y, X, T, seg_dir, volume):
    up = 4
    label = np.zeros((Y, X, 1, T), dtype='int16')

    # Fetch the corresponding seg contour file for each slice
    contours = loadmat(seg_dir, struct_as_record=False)['SEG'][0][0]
    # Labels NOTE: print(contours.keys())
    lv_endo = 1
    lv_epi = 2
    rv_endo = 3
    rv_epi = 4
    # la = 4
    # ra = 5

    # Fill the contours in order:
    #   1. LV endocardium
    #   2. LV epicardium
    #   3. RV endocardium
    #   4. RV epicardium
    #   5. Left Atrium
    #   6. Right Atrium

    ordered_contours = []

    if len(contours.EpiX) > 0:
        x = contours.EpiX[:, np.newaxis, :]
        y = contours.EpiY[:, np.newaxis, :]
        ordered_contours += [(np.concatenate((y, x), axis=1), lv_epi)]

    if len(contours.EndoX) > 0:
        x = contours.EndoX[:, np.newaxis, :]
        y = contours.EndoY[:, np.newaxis, :]
        ordered_contours += [(np.concatenate((y, x), axis=1), lv_endo)]

    if len(contours.RVEpiX) > 0:
        x = contours.RVEpiX[:, np.newaxis, :]
        y = contours.RVEpiY[:, np.newaxis, :]
        ordered_contours += [(np.concatenate((y, x), axis=1), rv_epi)]

    if len(contours.RVEndoX) > 0:
        x = contours.RVEndoX[:, np.newaxis, :]
        y = contours.RVEndoY[:, np.newaxis, :]
        ordered_contours += [(np.concatenate((y, x), axis=1), rv_endo)]

    lab_up = np.zeros((Y * up, X * up, T))
    for t in range(T):
        img_temp = np.zeros((Y * up, X * up))
        for c, l in ordered_contours:
            # c -> [n, (y, x), T]
            coord = c[..., t]
            if np.isnan(coord).any(): continue
            coord = np.round(coord * up).astype(np.int)
            cv2.fillPoly(img_temp, [coord], l)
        lab_up[..., t] = img_temp

    if label[:, :, 0].shape[0] == lab_up[::up, ::up].shape[0] and \
            label[:, :, 0].shape[1] == lab_up[::up, ::up].shape[1]:
        label[:, :, 0] = lab_up[::up, ::up]
    elif label[:, :, 0].shape[0] == lab_up[::up, ::up].shape[1] and \
            label[:, :, 0].shape[1] == lab_up[::up, ::up].shape[0]:
        label[:, :, 0] = lab_up[::up, ::up].transpose((1, 0, 2))

    if label.shape[0] != volume.shape[0] or label.shape[1] != volume.shape[1]:
        raise Exception('Image and contorus have different size!')
    else:
        return label


def get_nifti_CAP(source_dir, dest_dir):
    """
        Convert the files of image and mask to NIfTIs from the sorted CAP-cases folders
    """
    patient_IDs = glob(os.path.join(source_dir, '*'))
    for patient_ID in tqdm(patient_IDs):
        categories = [i for i in os.listdir(patient_ID) if '.seg' not in i]
        # create folder for storing nifti files (volumes and contours)
        dest_ID_dir = os.path.join(dest_dir, patient_ID.split('/')[-1])
        if not os.path.exists(dest_ID_dir):
            os.makedirs(dest_ID_dir)
        for category in categories:
            # get all dicom sequences from source_ID_dir
            slices = natsorted(os.listdir(os.path.join(patient_ID, category)))
            for z, slice in enumerate(slices):
                # change dicom file's name to its SOPInstanceUID
                sequence_dirs = natsorted(glob(os.path.join(patient_ID, category, slice, '*.dcm')))
                #for sequence_dir in sequence_dirs:
                #    dicom_file = pydicom.dcmread(sequence_dir)
                #    if hasattr(dicom_file, 'SOPInstanceUID'):
                #        SOPInstanceUID = dicom_file.SOPInstanceUID
                #    else:
                #        raise Exception(f"\rNo Series Instance UID in {patient_ID}'s dicom series\n")
                #    new_name = os.path.join('/', *sequence_dir.split('/')[:-1], SOPInstanceUID + '.dcm')
                #    os.rename(sequence_dir, new_name)
                sequence_dirs = natsorted(glob(os.path.join(patient_ID, category, slice, '*.dcm')))
                print(sequence_dirs)
                print(patient_ID)
                print(category)
                print(slice)
                # convert dicom sequences to nifti volume
                data_nifti_image = convert_dicom_to_nifti(sequence_dirs, single_slice=True)
                Y, X, _, T, _, volume, affine, hdr = data_nifti_image
                if z == 0:
                    image_nifti = volume
                    raw_hdr = hdr
                    raw_affine = affine
                else:
                    if image_nifti.shape[0] == volume.shape[0] and \
                            image_nifti.shape[1] == volume.shape[1]:
                        image_nifti = np.append(image_nifti, volume, axis=2)
                    elif image_nifti.shape[0] == volume.shape[1] and \
                            image_nifti.shape[1] == volume.shape[0]:
                        image_nifti = np.append(image_nifti, volume.transpose((1, 0, 2, 3)), axis=2)
                    else:
                        raise Exception(
                            f"\rIn {slice} of {patient_ID}, the size of image is not consistent through sequences"
                        )
                # Convert contours to nifti
                VIEW = 'LA' if 'long' in slice else 'SA'
                NUM = slice.split('_')[-1]
                seg_dir = glob(os.path.join(patient_ID, "*" + VIEW + "*" + NUM + "*.seg"))[0]
                contour = save_contours_CAP(Y, X, T, seg_dir, volume)
                if z == 0:
                    contour_nifti = contour
                else:
                    if contour_nifti.shape[0] == contour.shape[0] and \
                            contour_nifti.shape[1] == contour.shape[1]:
                        contour_nifti = np.append(contour_nifti, contour, axis=2)
                    elif contour_nifti.shape[0] == contour.shape[1] and \
                            contour_nifti.shape[1] == contour.shape[0]:
                        contour_nifti = np.append(contour_nifti, contour.transpose((1, 0, 2, 3)), axis=2)
                    else:
                        raise Exception(
                            f"\rIn {slice} of {patient_ID}, the size of contour is not consistent through sequences"
                        )
            # save image and contour nifti files
            image_nifti = nib.nifti1.Nifti1Image(image_nifti, raw_affine)
            contour_nifti = nib.nifti1.Nifti1Image(contour_nifti, raw_affine)
            for k, v in raw_hdr.items():
                if k in image_nifti.header:
                    image_nifti.header[k] = v
            for k, v in raw_hdr.items():
                if k in contour_nifti.header:
                    contour_nifti.header[k] = v
            image_nifti.header['datatype'] = data_type_codes.code[np.float32]
            dest_image_dir = os.path.join(dest_ID_dir, f'{category}_image.nii.gz')
            image_nifti.to_filename(dest_image_dir)
            contour_nifti.header['datatype'] = data_type_codes.code[np.int8]
            dest_contour_nifti = os.path.join(dest_ID_dir, f'{category}_contour.nii.gz')
            contour_nifti.to_filename(dest_contour_nifti)


def sort_folders(source_dir, dest_dir):
    lookup_table = pd.read_excel('VenusSliceLabels.xlsx', header=0, index_col=[0, 1], dtype=str)
    lookup_table.sort_index(level=[0, 1], inplace=True)
    prepost_column = ['preop', 'postop'] * (len(lookup_table) // 2)
    # study in an earlier date is 'pre-operation' otherwise 'post-operation'
    lookup_table['PrePost'] = prepost_column

    for patient_ID, study_ID in tqdm(lookup_table.index.tolist()):
        pre_or_post = lookup_table.at[(patient_ID, study_ID), 'PrePost']
        # copy the series folders to a view folder designated in the VenusSliceLabels.xlsx
        views = lookup_table.loc[patient_ID, study_ID].drop(['MRA', 'PrePost']).dropna()
        for view, num in zip(views.index.tolist(), views.values.tolist()):
            if ',' in num:
                nums = num.split(',')
            else:
                nums = [num]
            for n in nums:
                series_ID = 'unnamed_' + n
                source_dir_ = os.path.join(source_dir, patient_ID, patient_ID + '_' + str(study_ID), series_ID)
                dest_dir_ = os.path.join(
                    dest_dir, pre_or_post, patient_ID, patient_ID + '_' + str(study_ID), view, series_ID
                )
                if os.path.exists(dest_dir_):
                    shutil.rmtree(dest_dir_)
                shutil.copytree(source_dir_, dest_dir_)

        # then copy cvi42wsx files
        source_dir_ = glob(os.path.join(source_dir, patient_ID, '*' + str(study_ID) + '*.cvi42wsx'))[0]
        dest_dir_ = os.path.join(dest_dir, pre_or_post, patient_ID, source_dir_.split('/')[-1])
        shutil.copy(source_dir_, dest_dir_)


def sort_folders_CAP(source_dir, dest_dir):
    """
        Cleans and sorts the CAP_cases folder (source_dir) by the labels of image views,
        and saves a copy of the CAP_cases folder (dest_dir).
    """
    views_list = pd.read_excel("CAPSliceLabels.xlsx", header=0, index_col=0)
    for case_id in views_list.index:
        # sort the long axis first
        for slice in views_list.loc[case_id].dropna().index:
            NUM = slice.split('_')[-1]
            # find the folder of image dicoms
            dicom_source_dir = os.path.join(source_dir, case_id, slice.strip(' '))
            # find the corresponding segmentation masks
            contour_source_dir = None
            for f in glob(os.path.join(source_dir, case_id, '*.seg')):
                # exclude atria for now
                valid_seg = re.findall(f'(?![lL][aA][rR][aA])([lL][aA]_{NUM}_[bB][wW].seg)', f)
                if len(valid_seg) > 0:
                    contour_source_dir = f
                    break
            if contour_source_dir is None:
                print(f"\rNo coressponding contour for {slice} of {case_id}")
                continue
            view = views_list.loc[case_id, slice]
            dicom_dest_dir = os.path.join(dest_dir, case_id, view, slice)
            contour_dest_dir = os.path.join(dest_dir, case_id, contour_source_dir.split('/')[-1])
            shutil.copytree(dicom_source_dir, dicom_dest_dir)
            shutil.copy(contour_source_dir, contour_dest_dir)

        # sort the short axis now
        saxs_list = [i for i in os.listdir(os.path.join(source_dir, case_id)) if 'short' in i]
        for sax in saxs_list:
            NUM = sax.split('_')[-1]
            # find the folder of image dicoms
            dicom_source_dir = os.path.join(source_dir, case_id, "images", sax)
            # find the corresponding segmentation masks
            contour_source_dir = None
            for f in glob(os.path.join(source_dir, case_id, '*.seg')):
                valid_seg = re.findall(f'(?![lL][aA][rR][aA])([sS][aA]_{NUM}_[bB][wW].seg)', f)
                if len(valid_seg) > 0:
                    contour_source_dir = f
                    break
            if contour_source_dir is None:
                print(f"\rNo coressponding contour for {sax} of {case_id}")
                continue
            dicom_dest_dir = os.path.join(dest_dir, case_id, "SAX", sax)
            contour_dest_dir = os.path.join(dest_dir, case_id, contour_source_dir.split('/')[-1])
            shutil.copytree(dicom_source_dir, dicom_dest_dir)
            shutil.copy(contour_source_dir, contour_dest_dir)
