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
from data2D_reg import load_scan, get_pixels_hu

# Some constants
INPUT_FOLDER = '../input(backup)/CEFIESTA&GT/CEFIESTA'
INPUT_FOLDER_GT = '../input(backup)/CEFIESTA&GT/ground_truth'
image_rows = int(512)
image_cols = int(512)
image_depth = 1

patients = os.listdir(INPUT_FOLDER)
patients_gt = os.listdir(INPUT_FOLDER_GT)

patients.sort()
patients_gt.sort()

TRAIN_NUM = int((len(patients) - 1) * 0.1)
TEST_NUM = 15
print('-' * 30)
print('Creating training images...')
print('-' * 30)

for i in range(1, TRAIN_NUM + 1):
    patient_gt = load_scan(INPUT_FOLDER_GT + '/' + patients_gt[i])
    patient_pixels_gt = get_pixels_hu(patient_gt)
    for k in range(0, patient_pixels_gt.shape[0]):
        mask_df = patient_pixels_gt[k]
        if np.amax(mask_df) < 1:
            print(INPUT_FOLDER + '/' + patients[i])
            print(k + 1)

print('-' * 30)
print('Creating test images...')
print('-' * 30)

for i in range(TRAIN_NUM + 1, TEST_NUM + 1):
    patient_gt = load_scan(INPUT_FOLDER_GT + '/' + patients_gt[i])
    patient_pixels_gt = get_pixels_hu(patient_gt)
    for k in range(0, patient_pixels_gt.shape[0]):
        mask_df = patient_pixels_gt[k]
        if np.amax(mask_df) < 1:
            print(INPUT_FOLDER + '/' + patients[i])
            print(k + 1)