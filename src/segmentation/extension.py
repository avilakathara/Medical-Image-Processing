from pathlib import Path
import numpy as np
import SimpleITK as sitk
import os
import napari
import json
import nrrd
import tkinter as tk
from tkinter import filedialog
from segmentation import *
window = tk.Tk()
window.withdraw()
folder_path = filedialog.askdirectory()

window.destroy()



# INITIATE NAPARI
# viewer = napari.Viewer()

# CHOOSE ORGAN
# 1=BrainStem,2=Chiasm,3=Mandible,4=OpticNerve_L,5=OpticNerve_R,6=Parotid_L,7=Parotid_R,8=Submandibular_L,9=Submandibular_R)
chosen_organ = 3

# LOAD IMAGE
# img = np.load(folder_path + str("/") + os.listdir(folder_path)[0])
img_ai = np.load(folder_path + str("/") + os.listdir(folder_path)[1])

# LOAD PREDICTIONS
predictions = []
for i in range(2,8):
    prediction = np.load(folder_path + str("/") + os.listdir(folder_path)[i])
    prediction[prediction != chosen_organ] = 0
    prediction[prediction != 0] = 1
    predictions.append(prediction.astype(int))
predictions = np.array(predictions)
# LOAD GROUND TRUTH
# structures = np.load(folder_path + str("/") + os.listdir(folder_path)[8])
ground_truth = np.load(folder_path + str("/") + os.listdir(folder_path)[9])
ground_truth[ground_truth != chosen_organ] = 0
ground_truth[ground_truth != 0] = 1
ground_truth = ground_truth.astype(int)


approach = 3

# APPROACH 1
if approach == 1:
    avr_prediction = np.around(np.sum(predictions,axis=0)/6).astype(int)
    contours = []
    for slice in avr_prediction:
        contours.append(create_contour(slice))
    contours = np.array(contours).astype(int)
    seed_points = convert_to_labels(contours)

# APPROACH 2
if approach == 2:
    avr_prediction = np.sum(predictions,axis=0).astype(int)
    avr_prediction[avr_prediction != 6] = 0
    best_slice = 0
    best_count = 0
    for index,slice in enumerate(avr_prediction):
        if np.count_nonzero(slice) > best_count:
            best_count = np.count_nonzero(slice)
            best_slice = index
    seed_points = np.zeros(avr_prediction.shape).astype(int)
    seed_points[best_slice] = avr_prediction[best_slice]
    seed_points[best_slice][seed_points[best_slice] != 0] = 1
    seed_points[best_slice][seed_points[best_slice] == 0] = 2


# APPROACH 3
# unanimous votes
if approach == 3:
    sum_predictions = np.sum(predictions,axis=0).astype(int)
    seed_points = np.zeros(sum_predictions.shape)
    seed_points[sum_predictions == 0] = 2
    seed_points[sum_predictions == 6] = 1


# viewer.add_image(img, name="CT_SCAN", colormap="gray", interpolation2d="bicubic")
# viewer.add_image(img_ai, name="CT_SCAN AI", colormap="gray", interpolation2d="bicubic")
# viewer.add_labels(seed_points.astype(int), name="seed points")


#
# segmentation, prob = segment(img_ai,seed_points)
# viewer.add_labels(segmentation,name="segmentation")
#


def save_all_as_numpy():
    input_path = Path(r'C:\Users\Bram\Documents\RP\miccai-test-mc-preds')
    output_path = Path(r'C:\Users\Bram\Documents\RP\miccai_data_numpy')

    for p in input_path.rglob("img.nrrd"):

        patient_base_path = p.parent
        patient_id = patient_base_path.stem

        for output in output_path.rglob("img.npy"):
            if patient_id != output.parent.stem:
                continue

            img_path = patient_base_path.joinpath("img.nrrd")
            img = sitk.ReadImage(img_path)
            img_arr = sitk.GetArrayFromImage(img)
            np.save(output.parent.joinpath("img_ai.npy"), img_arr)


            img_path = patient_base_path.joinpath("mask.nrrd")
            img = sitk.ReadImage(img_path)
            img_arr = sitk.GetArrayFromImage(img)
            np.save(output.parent.joinpath("structures_ai.npy"), img_arr)


            for i in range(1,7):
                img_path = patient_base_path.joinpath("maskpred_"+str(i)+".nrrd")
                img = sitk.ReadImage(img_path)
                img_arr = sitk.GetArrayFromImage(img)
                np.save(output.parent.joinpath("maskpred_"+str(i)+".npy"), img_arr)


# napari.run()

