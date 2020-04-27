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

# Some constants
INPUT_FOLDER = '../input/CEFIESTA&GT/CEFIESTA'
INPUT_FOLDER_GT = '../input/CEFIESTA&GT/ground_truth'
image_rows = int(512)
image_cols = int(512)
image_depth = 16

# Flag of data argument
argument = 0

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


def create_train_data():
    # TRAIN_NUM = int((len(patients) - 1) * 0.7)
    TRAIN_NUM = int((len(patients) - 1) * 0.1)

    if argument == 1:
        # MAX of slices per patient: 216
        total = 0
        for k in range(0, TRAIN_NUM):
            images = os.listdir(INPUT_FOLDER + '/' + patients[k + 1])
            total += int(np.floor(len(images) / 8))
        total += 1

        # total = int(TRAIN_NUM * 27)

        imgs = np.ndarray((total, image_depth, image_rows, image_cols), dtype=np.int16)
        imgs_mask = np.ndarray((total, image_depth, image_rows, image_cols), dtype=np.int16)

        imgs_temp = np.ndarray((total, image_depth//2, image_rows, image_cols), dtype=np.int16)
        imgs_mask_temp = np.ndarray((total, image_depth//2, image_rows, image_cols), dtype=np.int16)

        num = 0
        print('-' * 30)
        print('Creating training images...')
        print('-' * 30)

        for i in range(0, TRAIN_NUM):
            j = 0
            patient = load_scan(INPUT_FOLDER + '/' + patients[i + 1])
            patient_pixels = get_pixels_hu(patient)
            count = total
            for k in range(0, patient_pixels.shape[0]):
                imgs_temp[num, j] = patient_pixels[k]
                j += 1
                if j % (image_depth/2) == 0:
                    j = 0
                    num += 1
                    if (num % 100) == 0:
                        print('Done: {0}/{1} 3d images'.format(num, count))

        for x in range(0, imgs_temp.shape[0] - 1):
            imgs[x] = np.append(imgs_temp[x], imgs_temp[x+1], axis=0)

        print('Loading of train data done.')

        print('-' * 30)
        print('Creating training masks...')
        print('-' * 30)

        num = 0

        for i in range(0, TRAIN_NUM):
            j = 0
            patient_gt = load_scan(INPUT_FOLDER_GT + '/' + patients_gt[i + 1])
            patient_pixels_gt = get_pixels_hu(patient_gt)
            count = total
            for k in range(0, patient_pixels_gt.shape[0]):
                imgs_mask_temp[num, j] = patient_pixels_gt[k]
                j += 1
                if j % (image_depth / 2) == 0:
                    j = 0
                    num += 1
                    if (num % 100) == 0:
                        print('Done: {0}/{1} 3d images'.format(num, count))

        for x in range(0, imgs_mask_temp.shape[0] - 1):
            imgs_mask[x] = np.append(imgs_mask_temp[x], imgs_mask_temp[x + 1], axis=0)

        print('Loading of masks done.')

        imgs_mask = preprocess(imgs_mask)
        imgs = preprocess(imgs)

        print('Preprocessing of masks done.')

        np.save('imgs_train_3D.npy', imgs)
        np.save('imgs_mask_train_3D.npy', imgs_mask)

        imgs = preprocess_squeeze(imgs)
        imgs_mask = preprocess_squeeze(imgs_mask)

        print('Saving to .npy files done.')

    elif argument == 0:
        # MAX of slices per patient: 216
        total = 0
        for k in range(0, TRAIN_NUM):
            images = os.listdir(INPUT_FOLDER + '/' + patients[k + 1])
            total += len(images)
        total = int(np.ceil(total / image_depth))

        # total = int(TRAIN_NUM * 27)

        imgs = np.ndarray((total, image_depth, image_rows, image_cols), dtype=np.int16)
        imgs_mask = np.ndarray((total, image_depth, image_rows, image_cols), dtype=np.int16)

        print('-' * 30)
        print('Creating training images...')
        print('-' * 30)

        num = 0
        j = 0
        for i in range(0, TRAIN_NUM):
            patient = load_scan(INPUT_FOLDER + '/' + patients[i + 1])
            patient_pixels = get_pixels_hu(patient)
            count = total
            for k in range(0, patient_pixels.shape[0]):
                imgs[num, j] = patient_pixels[k]
                j += 1
                if j % image_depth == 0:
                    j = 0
                    num += 1
                    if (num % 100) == 0:
                        print('Done: {0}/{1} 3d images'.format(num, count))

        print('Loading of train data done.')

        print('-' * 30)
        print('Creating training masks...')
        print('-' * 30)

        num = 0
        j = 0
        for i in range(0, TRAIN_NUM):
            patient_gt = load_scan(INPUT_FOLDER_GT + '/' + patients_gt[i + 1])
            patient_pixels_gt = get_pixels_hu(patient_gt)
            count = total
            for k in range(0, patient_pixels_gt.shape[0]):
                imgs_mask[num, j] = patient_pixels_gt[k]
                j += 1
                if j % image_depth == 0:
                    j = 0
                    num += 1
                    if (num % 100) == 0:
                        print('Done: {0}/{1} 3d images'.format(num, count))

        print('Loading of masks done.')

        imgs_mask = preprocess(imgs_mask)
        imgs = preprocess(imgs)

        print('Preprocessing of masks done.')

        np.save('imgs_train_3D.npy', imgs)
        np.save('imgs_mask_train_3D.npy', imgs_mask)

        imgs = preprocess_squeeze(imgs)
        imgs_mask = preprocess_squeeze(imgs_mask)

        print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train_3D.npy')
    imgs_mask_train = np.load('imgs_mask_train_3D.npy')
    return imgs_train, imgs_mask_train


def create_test_data():
    # TRAIN_NUM = int((len(patients) - 1) * 0.7)
    TRAIN_NUM = int((len(patients) - 1) * 0.1)
    # TEST_NUM = 91
    TEST_NUM = 15

    if argument == 1:
        total = 0
        for k in range(TRAIN_NUM, TEST_NUM):
            images = os.listdir(INPUT_FOLDER + '/' + patients[k + 1])
            total += len(images)
            # total += int(np.floor((len(images) - 2) / (image_depth - 2)))
        total = int(np.ceil((total - 2) / (image_depth - 2)))

        imgs = np.ndarray((total, image_depth, image_rows, image_cols), dtype=np.int16)
        imgs_mask = np.ndarray((total, image_depth, image_rows, image_cols), dtype=np.int16)

        print('-' * 30)
        print('Creating test images...')
        print('-' * 30)

        num = 0
        j = 0
        for i in range(TRAIN_NUM, TEST_NUM):
            patient = load_scan(INPUT_FOLDER + '/' + patients[i + 1])
            patient_pixels = get_pixels_hu(patient)
            count = total
            for k in range(0, patient_pixels.shape[0]):
                imgs[num, j] = patient_pixels[k]
                j += 1
                if j % (image_depth - 1) == 0:
                    imgs[num + 1, 0] = patient_pixels[k]
                if j % image_depth == 0:
                    imgs[num + 1, 1] = patient_pixels[k]
                    j = 2
                    num += 1
                    print('Done: {0}/{1} test 3d images'.format(num, count))

        print('Loading of test data done.')

        print('-' * 30)
        print('Creating test masks...')
        print('-' * 30)

        num = 0
        j = 0
        for i in range(TRAIN_NUM, TEST_NUM):
            patient_gt = load_scan(INPUT_FOLDER_GT + '/' + patients_gt[i + 1])
            patient_pixels_gt = get_pixels_hu(patient_gt)
            count = total
            for k in range(0, patient_pixels_gt.shape[0]):
                imgs_mask[num, j] = patient_pixels_gt[k]
                j += 1
                if j % (image_depth - 1) == 0:
                    imgs_mask[num + 1, 0] = patient_pixels_gt[k]
                if j % image_depth == 0:
                    imgs_mask[num + 1, 1] = patient_pixels_gt[k]
                    j = 2
                    num += 1
                    print('Done: {0}/{1} test mask 3d images'.format(num, count))

        print('Loading of test masks done.')

        imgs = preprocess(imgs)
        imgs_mask = preprocess(imgs_mask)

        np.save('imgs_test_3D.npy', imgs)
        np.save('imgs_mask_test_3D.npy', imgs_mask)

        imgs = preprocess_squeeze(imgs)
        imgs_mask = preprocess_squeeze(imgs_mask)

        count_processed = 0
        pred_dir = 'test_preprocessed_3D'
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        for x in range(0, imgs.shape[0]):
            for y in range(0, imgs.shape[1]):
                io.imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs[x][y])
                count_processed += 1
                if (count_processed % 100) == 0:
                    print('Done: {0}/{1} test images'.format(count_processed, imgs.shape[0] * imgs.shape[1]))

        count_processed = 0
        pred_dir = 'test_mask_preprocessed_3D'
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        for x in range(0, imgs_mask.shape[0]):
            for y in range(0, imgs_mask.shape[1]):
                io.imsave(os.path.join(pred_dir, 'pre_processed_mask_' + str(count_processed) + '.png'), imgs_mask[x][y])
                count_processed += 1
                if (count_processed % 100) == 0:
                    print('Done: {0}/{1} test images'.format(count_processed, imgs_mask.shape[0] * imgs_mask.shape[1]))

        print('Saving to .npy files done.')

    elif argument == 0:
        total = 0
        for k in range(TRAIN_NUM, TEST_NUM):
            images = os.listdir(INPUT_FOLDER + '/' + patients[k + 1])
            total += len(images)
        total = int(np.ceil(total / image_depth))

        imgs = np.ndarray((total, image_depth, image_rows, image_cols), dtype=np.int16)
        imgs_mask = np.ndarray((total, image_depth, image_rows, image_cols), dtype=np.int16)

        print('-' * 30)
        print('Creating test images...')
        print('-' * 30)

        num = 0
        j = 0
        for i in range(TRAIN_NUM, TEST_NUM):
            patient = load_scan(INPUT_FOLDER + '/' + patients[i + 1])
            patient_pixels = get_pixels_hu(patient)
            count = total
            for k in range(0, patient_pixels.shape[0]):
                imgs[num, j] = patient_pixels[k]
                j += 1
                if j % image_depth == 0:
                    j = 0
                    num += 1
                    print('Done: {0}/{1} test 3d images'.format(num, count))

        print('Loading of test data done.')

        print('-' * 30)
        print('Creating test masks...')
        print('-' * 30)

        num = 0
        j = 0
        for i in range(TRAIN_NUM, TEST_NUM):
            patient_gt = load_scan(INPUT_FOLDER_GT + '/' + patients_gt[i + 1])
            patient_pixels_gt = get_pixels_hu(patient_gt)
            count = total
            for k in range(0, patient_pixels_gt.shape[0]):
                imgs_mask[num, j] = patient_pixels_gt[k]
                j += 1
                if j % image_depth == 0:
                    j = 0
                    num += 1
                    print('Done: {0}/{1} test mask 3d images'.format(num, count))

        print('Loading of test masks done.')

        imgs = preprocess(imgs)
        imgs_mask = preprocess(imgs_mask)

        np.save('imgs_test_3D.npy', imgs)
        np.save('imgs_mask_test_3D.npy', imgs_mask)

        imgs = preprocess_squeeze(imgs)
        imgs_mask = preprocess_squeeze(imgs_mask)

        count_processed = 0
        pred_dir = 'test_preprocessed_3D'
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        for x in range(0, imgs.shape[0]):
            for y in range(0, imgs.shape[1]):
                io.imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs[x][y])
                count_processed += 1
                if (count_processed % 100) == 0:
                    print('Done: {0}/{1} test images'.format(count_processed, imgs.shape[0] * imgs.shape[1]))

        count_processed = 0
        pred_dir = 'test_mask_preprocessed_3D'
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        for x in range(0, imgs_mask.shape[0]):
            for y in range(0, imgs_mask.shape[1]):
                io.imsave(os.path.join(pred_dir, 'pre_processed_mask_' + str(count_processed) + '.png'),
                          imgs_mask[x][y])
                count_processed += 1
                if (count_processed % 100) == 0:
                    print('Done: {0}/{1} test images'.format(count_processed, imgs_mask.shape[0] * imgs_mask.shape[1]))

        print('Saving to .npy files done.')

def load_test_data():
    imgs_test = np.load('imgs_test_3D.npy')
    return imgs_test


def preprocess(imgs):
    imgs = np.expand_dims(imgs, axis=4)
    print(' ---------------- preprocessed -----------------')
    return imgs


def preprocess_squeeze(imgs):
    imgs = np.squeeze(imgs, axis=4)
    print(' ---------------- preprocessed squeezed -----------------')
    return imgs


if __name__ == '__main__':
    create_train_data()
    create_test_data()
