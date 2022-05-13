from inspect import FrameInfo
import os
import shutil
from glob import glob
from cv2 import FileStorage
from natsort.natsort import natsorted

import nibabel as nib
from nibabel.nifti1 import unit_codes, xform_codes, data_type_codes

import pandas as pd
import numpy as np

from skimage import measure

import utils


def remove_small_cc(binary, thres=10):
    """ Remove small connected component in the foreground. """
    cc, n_cc = measure.label(binary, return_num=True)
    binary2 = np.copy(binary)
    for n in range(1, n_cc + 1):
        area = np.sum(cc == n)
        if area < thres:
            binary2[cc == n] = 0
    return binary2


def get_largest_cc(binary):
    """ Get the largest connected component in the foreground. """
    cc, n_cc = measure.label(binary, return_num=True)
    max_n = -1
    max_area = 0
    for n in range(1, n_cc + 1):
        area = np.sum(cc == n)
        if area > max_area:
            max_area = area
            max_n = n
    largest_cc = (cc == max_n)
    return largest_cc


def clean_label(label_data, all_frame=False):
    '''
        Clean the labels with
        a) small connected components (myo);
        b) shift one pixel left and up respectively,
        ensuring labels overlaying the images seemlessly (CAP only)
        c) largest connected components (blood pool)
    '''

    # Label class in the segmentation
    label = {'BG': 0, 'LV cavity': 1, 'LV Myo': 2, 'RV cavity': 3, 'RV Myo': 4}

    if all_frame:
        for frame in range(label_data.shape[-1]):
            # a)
            lv_myo = (label_data[..., frame] == label['LV Myo']).astype(int)
            lv_myo = remove_small_cc(lv_myo)
            rv_myo = (label_data[..., frame] == label['RV Myo']).astype(int)
            rv_myo = remove_small_cc(rv_myo)

            # c)
            lv_bp = (label_data[..., frame] == label['LV cavity']).astype(int)
            lv_bp = get_largest_cc(lv_bp)
            rv_bp = (label_data[..., frame] == label['RV cavity']).astype(int)
            rv_bp = get_largest_cc(rv_bp)

            label_data[..., frame] = label['LV Myo'] * lv_myo + label['RV Myo'] * rv_myo + \
                                     label['LV cavity'] * lv_bp + label['RV cavity'] * rv_bp

    else:
        # a)
        lv_myo = (label_data == label['LV Myo']).astype(int)
        lv_myo = remove_small_cc(lv_myo)
        rv_myo = (label_data == label['RV Myo']).astype(int)
        rv_myo = remove_small_cc(rv_myo)

        # c)
        lv_bp = (label_data == label['LV cavity']).astype(int)
        lv_bp = get_largest_cc(lv_bp)
        rv_bp = (label_data == label['RV cavity']).astype(int)
        rv_bp = get_largest_cc(rv_bp)

        label_data = label['LV Myo'] * lv_myo + label['RV Myo'] * rv_myo + \
                     label['LV cavity'] * lv_bp + label['RV cavity'] * rv_bp

    # b)
    label_data = np.roll(label_data, (-1, -1), (0, 1))

    return label_data


def slice_sort(category, label_data, image_data, affine, new_hdr, is_train=True, **args):
    '''
        data_shape is (Y, X, Z, T) and should be converted into (Y, X, Z)
    '''
    case_id = args.get('case_id')
    # Find ED and ES frames
    frame_index = np.sum(label_data != 0, axis=(0, 1, 2))
    if not frame_index.any():
        raise ValueError(f"No valid label in {args.get('patient_id')}.")
    frame_index = [(index, volume) for index, volume in enumerate(frame_index)]
    frame_index = np.array(frame_index, dtype=[("index", int), ("volume", float)])
    frame_index = np.sort(frame_index, order="volume")[-2:]

    for (t, _), frame in zip(frame_index, ['ES', 'ED']):
        if category != 'SAX':
            slice_idx = (np.sum(label_data[..., t], axis=(0, 1)) != 0)
            image_data_ = image_data[:, :, slice_idx, t]
            label_data_ = label_data[:, :, slice_idx, t]
        else:
            # All slices from SAX one needed
            image_data_ = image_data[..., t]
            label_data_ = label_data[..., t]
        assert image_data_.shape == label_data_.shape
        label_data_ = clean_label(label_data_)
        # Save new NIfTI files complied with nnUNet requirements
        if is_train:
            dest_image_dir = os.path.join(temp_dir, category, 'imagesTr')
            dest_label_dir = os.path.join(temp_dir, category, 'labelsTr')
        else:
            dest_image_dir = os.path.join(temp_dir, category, frame, 'imagesTs')
            dest_label_dir = os.path.join(temp_dir, category, frame, 'labelsTs')
        if not os.path.exists(dest_image_dir):
            os.makedirs(dest_image_dir)
            os.makedirs(dest_label_dir)
        image_name = f'{case_id}_{frame}_0000.nii.gz'
        label_name = f'{case_id}_{frame}.nii.gz'
        dest_image_dir = os.path.join(dest_image_dir, image_name)
        dest_label_dir = os.path.join(dest_label_dir, label_name)

        image_nifti_ = nib.nifti1.Nifti1Image(image_data_, affine)
        label_nifti_ = nib.nifti1.Nifti1Image(label_data_, affine)
        for k, v in new_hdr.items():
            if k in image_nifti_.header:
                image_nifti_.header[k] = v
        for k, v in new_hdr.items():
            if k in label_nifti_.header:
                label_nifti_.header[k] = v
        nib.save(image_nifti_, dest_image_dir)
        nib.save(label_nifti_, dest_label_dir)


def nnunet_prep(**args):
    """Change directories of NIfTI for nnUNet training/testing"""

    categories = args.get('categories')

    for category in categories:
        cat_image_files = natsorted(
            glob(os.path.join(
                dest_dir, '*', '*' + category + '_image.nii.gz'
            ))
        )
        cat_label_files = natsorted(
            glob(os.path.join(
                dest_dir, '*', '*' + category + '_contour.nii.gz'
            ))
        )

        for source_image_dir, source_label_dir in zip(cat_image_files, cat_label_files):
            case_id = source_image_dir.split('/')[-2]
            # Read NIfTI files and get numpy data
            image_nifti = nib.load(source_image_dir)
            image_data = image_nifti.get_fdata()  # (Y, X, Z, T)
            label_nifti = nib.load(source_label_dir)
            label_data = label_nifti.get_fdata()
            affine = image_nifti.affine

            # Index of labels are,
            #   0. Background
            #   1. LV endocardium
            #   2. LV epicardium
            #   3. RV endocardium
            #   4. RV epicardium

            # Rewrite the header of nifti
            nifti_hdr = image_nifti.header
            nifti_hdr['pixdim'][4] = 0
            new_hdr = {
                'pixdim': nifti_hdr['pixdim'],
                'toffset': nifti_hdr['toffset'],
                'slice_duration': nifti_hdr['slice_duration'],
                'xyzt_units': unit_codes['mm'] | unit_codes['msec'],
                'qform_code': xform_codes['aligned'],
                'sform_code': xform_codes['scanner'],
                'datatype': data_type_codes.code[np.float32]
            }

            slice_sort(
                category,
                label_data,
                image_data,
                affine,
                new_hdr,
                is_train=True,
                patient_id=source_image_dir,
                case_id=case_id
            )


if __name__ == '__main__':

    root_dir = 'E:/CAP/CAP-FullAutomation/data/raw/Longitudinal Cases/FILTERED_DATA/'
    source_dir = 'E:/CAP/CAP-FullAutomation/data/processed/'
    dest_dir = 'E:/CAP/CAP-FullAutomation/data/final/'
    temp_dir = 'E:/CAP/CAP-FullAutomation/data/temp/'

    #if os.path.exists(source_dir):
    #    shutil.rmtree(source_dir)
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    # Get a new copy and sort folders
    utils.sort_folders_CAP(root_dir, source_dir)

    # Transform DICOMs & cvi42wsx files into NIfTIs
    utils.get_nifti_CAP(source_dir, dest_dir)

    categories = ['SAX', '2CH', '3CH', '4CH', 'RVOT', 'RVT']
    # categories = ['SAX']
    nnunet_prep(categories=categories)
