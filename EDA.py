import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import snowy

from distance import *

INPUT_FOLDER = '../input/CEFIESTA&GT/CEFIESTA'
INPUT_FOLDER_GT = '../input/CEFIESTA&GT/ground_truth'
image_rows = int(512)
image_cols = int(512)
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


TRAIN_NUM = int((len(patients) - 1) * 0.1)
total = 0
for k in range(0, TRAIN_NUM):
    images = os.listdir(INPUT_FOLDER + '/' + patients[k + 1])
    total += len(images)

udf_list = np.ndarray((total, image_rows, image_cols, 1), dtype=np.int16)
sdf_list = np.ndarray((total, image_rows, image_cols, 1), dtype=np.int16)

num = 0
for i in range(0, TRAIN_NUM):
    j = 0
    patient_gt = load_scan(INPUT_FOLDER_GT + '/' + patients_gt[i + 1])
    patient_pixels_gt = get_pixels_hu(patient_gt)
    for k in range(0, patient_pixels_gt.shape[0]):
        mask_df = patient_pixels_gt[k][:, :, np.newaxis]
        # Outside
        udf = generate_udf(mask_df != 0.0)
        # Outside - Inside
        sdf = generate_sdf(mask_df != 0.0)
        udf_list[num] = udf
        sdf_list[num] = sdf
        num += 1
    print('Done: {0}/{1} masks'.format(i, TRAIN_NUM))

plt.hist(udf_list.flatten(), bins=80, color='c')
plt.xlabel("Distance Field")
plt.ylabel("Frequency")
plt.show()

plt.hist(sdf_list.flatten(), bins=80, color='c')
plt.xlabel("Distance Field")
plt.ylabel("Frequency")
plt.show()
