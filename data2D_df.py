# import package
# linear algebra
import numpy as np
# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import pydicom
import os
import scipy.ndimage
import sys
import matplotlib.pyplot as plt

from skimage import measure, morphology, io
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from distance import *

# Some constants
INPUT_FOLDER = '../input/CEFIESTA&GT/CEFIESTA'
INPUT_FOLDER_GT = '../input/CEFIESTA&GT/ground_truth'
image_rows = int(512)
image_cols = int(512)
image_depth = 1

patients = os.listdir(INPUT_FOLDER)
patients_gt = os.listdir(INPUT_FOLDER_GT)

patients.sort()
patients_gt.sort()

# Load the scans in given folder path
def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = 0
        slope = 1
        if hasattr(slices[slice_number], 'RescaleIntercept'):
            intercept = slices[slice_number].RescaleIntercept
        if hasattr(slices[slice_number], 'RescaleSlope'):
            slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def df_class(image):
    result = np.zeros((image.shape[0], image.shape[1]), dtype=np.int16)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if image[i, j] < -1:
                result[i, j] = 1
            elif image[i, j] < 0:
                result[i, j] = 0
            elif image[i, j] < 100:
                result[i, j] = 2
            elif image[i, j] < 200:
                result[i, j] = 3
            elif image[i, j] < 300:
                result[i, j] = 4
            else:
                result[i, j] = 5
    return result


def create_train_data():
    # TRAIN_NUM = int((len(patients) - 1) * 0.7)
    TRAIN_NUM = int((len(patients) - 1) * 0.1)

    imgs_temp = np.ndarray((0, image_rows, image_cols), dtype=np.int16)
    imgs_mask_temp = np.ndarray((0, image_rows, image_cols), dtype=np.int16)

    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)

    for i in range(1, TRAIN_NUM + 1):
        patient = load_scan(INPUT_FOLDER + '/' + patients[i])
        patient_pixels = get_pixels_hu(patient)
        imgs_temp = np.concatenate((imgs_temp, patient_pixels), axis=0)
        print('Done: {0}/{1} 2d images'.format(i, TRAIN_NUM))

    imgs = imgs_temp

    print('Loading of train data done.')

    print('-' * 30)
    print('Creating training masks...')
    print('-' * 30)

    for i in range(1, TRAIN_NUM + 1):
        patient_gt = load_scan(INPUT_FOLDER_GT + '/' + patients_gt[i])
        patient_pixels_gt = get_pixels_hu(patient_gt)
        for k in range(0, patient_pixels_gt.shape[0]):
            mask_df = patient_pixels_gt[k][:, :, np.newaxis]
            sdf = generate_sdf(mask_df != 0)
            sdf = sdf[:, :, 0]
            sdf_class = df_class(sdf)
            sdf_class = sdf_class[np.newaxis, :, :]
            imgs_mask_temp = np.concatenate((imgs_mask_temp, sdf_class), axis=0)
        print('Done: {0}/{1} mask 2d images'.format(i, TRAIN_NUM))

    imgs_mask = imgs_mask_temp

    print('Loading of train masks done.')

    imgs = preprocess(imgs)
    imgs_mask = preprocess(imgs_mask)

    print('Preprocessing of masks done.')

    np.save('imgs_train_df.npy', imgs)
    np.save('imgs_mask_train_df.npy', imgs_mask)

    imgs = preprocess_squeeze(imgs)
    imgs_mask = preprocess_squeeze(imgs_mask)

    # count_processed = 0
    # pred_dir = 'train_preprocessed'
    # if not os.path.exists(pred_dir):
    #     os.mkdir(pred_dir)
    # for x in range(0, imgs.shape[0]):
    #     io.imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs[x])
    #     count_processed += 1
    #     if (count_processed % 100) == 0:
    #         print('Done: {0}/{1} train images'.format(count_processed, imgs.shape[0]))
    #
    # count_processed = 0
    # pred_dir = 'mask_preprocessed'
    # if not os.path.exists(pred_dir):
    #     os.mkdir(pred_dir)
    # for x in range(0, imgs.shape[0]):
    #     io.imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs_mask[x])
    #     count_processed += 1
    #     if (count_processed % 100) == 0:
    #         print('Done: {0}/{1} train images'.format(count_processed, imgs.shape[0]))

    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train_df.npy')
    imgs_mask_train = np.load('imgs_mask_train_df.npy')
    return imgs_train, imgs_mask_train


def create_test_data():
    # TRAIN_NUM = int((len(patients) - 1) * 0.7)
    TRAIN_NUM = int((len(patients) - 1) * 0.1)
    # TEST_NUM = 91
    TEST_NUM = 15

    imgs_temp = np.ndarray((0, image_rows, image_cols), dtype=np.int16)
    imgs_mask_temp = np.ndarray((0, image_rows, image_cols), dtype=np.int16)

    print('-' * 30)
    print('Creating test images...')
    print('-' * 30)

    for i in range(TRAIN_NUM + 1, TEST_NUM + 1):
        patient = load_scan(INPUT_FOLDER + '/' + patients[i])
        patient_pixels = get_pixels_hu(patient)
        imgs_temp = np.concatenate((imgs_temp, patient_pixels), axis=0)
        print('Done: {0}/{1} 2d images'.format(i - TRAIN_NUM - 1, TEST_NUM - TRAIN_NUM))

    imgs = imgs_temp

    print('Loading of test data done.')

    print('-' * 30)
    print('Creating test masks...')
    print('-' * 30)

    for i in range(TRAIN_NUM + 1, TEST_NUM + 1):
        patient_gt = load_scan(INPUT_FOLDER_GT + '/' + patients_gt[i])
        patient_pixels_gt = get_pixels_hu(patient_gt)
        for k in range(0, patient_pixels_gt.shape[0]):
            mask_df = patient_pixels_gt[k][:, :, np.newaxis]
            sdf = generate_sdf(mask_df != 0)
            sdf = sdf[:, :, 0]
            sdf_class = df_class(sdf)
            sdf_class = sdf_class[np.newaxis, :, :]
            imgs_mask_temp = np.concatenate((imgs_mask_temp, sdf_class), axis=0)
        print('Done: {0}/{1} 2d images'.format(i - TRAIN_NUM - 1, TEST_NUM - TRAIN_NUM))

    imgs_mask = imgs_mask_temp

    print('Loading of test masks done.')

    imgs = preprocess(imgs)
    imgs_mask = preprocess(imgs_mask)

    np.save('imgs_test_df.npy', imgs)
    np.save('imgs_mask_test_df.npy', imgs_mask)

    imgs = preprocess_squeeze(imgs)
    imgs_mask = preprocess_squeeze(imgs_mask)

    count_processed = 0
    pred_dir = 'test_preprocessed_df'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for x in range(0, imgs.shape[0]):
        io.imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs[x])
        count_processed += 1
        if (count_processed % 100) == 0:
            print('Done: {0}/{1} test images'.format(count_processed, imgs.shape[0]))

    count_processed = 0
    pred_dir = 'test_mask_preprocessed_df'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for x in range(0, imgs.shape[0]):
        io.imsave(os.path.join(pred_dir, 'pre_processed_mask_' + str(count_processed) + '.png'), imgs_mask[x])
        count_processed += 1
        if (count_processed % 100) == 0:
            print('Done: {0}/{1} test images'.format(count_processed, imgs.shape[0]))

    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test_df.npy')
    return imgs_test


def preprocess(imgs):
    imgs = np.expand_dims(imgs, axis=3)
    print(' ---------------- preprocessed -----------------')
    return imgs


def preprocess_squeeze(imgs):
    imgs = np.squeeze(imgs, axis=3)
    print(' ---------------- preprocessed squeezed -----------------')
    return imgs


if __name__ == '__main__':
    # create_train_data()
    create_test_data()


# imgs, imgs_mask = create_train_data()
# print(np.amax(imgs))
# print(np.amax(imgs_mask))
#
# first_patient = load_scan(INPUT_FOLDER + '/' + patients[1])
# second_patient = load_scan(INPUT_FOLDER + '/' + patients[2])
# first_patient_pixels = get_pixels_hu(first_patient)
# second_patient_pixels = get_pixels_hu(second_patient)
# pixels = np.concatenate((first_patient_pixels, second_patient_pixels), axis=0)
#
# first_patient_gt = load_scan(INPUT_FOLDER + '/' + patients_gt[1])
