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
from skimage.morphology import binary_dilation
from skimage.morphology import binary_erosion
from skimage.segmentation import flood

window = tk.Tk()
window.withdraw()
folder_path = filedialog.askdirectory()

window.destroy()
def dice_coefficient(mask1, mask2):
    intersection = np.sum(mask1 * mask2)
    total = np.sum(mask1) + np.sum(mask2)
    dice = (2.0 * intersection) / total
    return dice


# INITIATE NAPARI
viewer = napari.Viewer()

# CHOOSE ORGAN
# 1=BrainStem,2=Chiasm,3=Mandible,4=OpticNerve_L,5=OpticNerve_R,6=Parotid_L,7=Parotid_R,8=Submandibular_L,9=Submandibular_R)
chosen_organ = 3

# LOAD IMAGE
# img = np.load(folder_path + str("/") + os.listdir(folder_path)[0])
img_ai = np.load(folder_path + str("/") + os.listdir(folder_path)[1])

# LOAD PREDICTIONS
global predictions
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


approach = 1

# APPROACH 1
# majority vote
if approach == 1:
    maj_vote = np.around(np.sum(predictions,axis=0)/len(predictions)).astype(int)
    seed_points = np.zeros(maj_vote.shape)
    for index,slice in enumerate(maj_vote):
        dilated = binary_dilation(slice,footprint=np.ones((2,2)))
        eroded = binary_erosion(slice,footprint=np.ones((2,2)))
        indices = np.argwhere(dilated == 0)[0]
        seed_points[index][flood(dilated,(indices[0],indices[1]))] = 2
        seed_points[index][eroded == 1] = 1

    # contours = []
    # for slice in maj_vote:
    #     contours.append(create_contour(slice))
    # contours = np.array(contours).astype(int)
    # seed_points = convert_to_labels(contours)

# APPROACH 2
# unanimous votes
if approach == 2:
    sum_pred = np.sum(predictions,axis=0).astype(int)
    seed_points = np.zeros(sum_pred.shape)
    seed_points[sum_pred == 0] = 2
    seed_points[sum_pred == 6] = 1

# APPROACH 3
# selected positive unanimous votes
if approach == 3:
    sum_pred = np.sum(predictions,axis=0).astype(int)
    seed_points = np.zeros(sum_pred.shape)
    thres = 0.9
    for index,slice in enumerate(sum_pred):
        if np.count_nonzero(slice) == 0:
            continue
        if np.count_nonzero(slice==6)/np.count_nonzero(slice) > thres:
            seed_points[index][slice==0] = 2
            seed_points[index][slice==6] = 1


# APPROACH 4
# use baseline seeds
if approach == 4:
    seed_points = convert_to_labels(automatic_contours(ground_truth))

# viewer.add_image(img, name="CT_SCAN", colormap="gray", interpolation2d="bicubic")
viewer.add_image(img_ai, name="CT_SCAN AI", colormap="gray", interpolation2d="bicubic")
viewer.add_labels(seed_points.astype(int), name="seed points")


print("img shape: ",img_ai.shape,", seed_points shape: ",seed_points.shape,", predictions shape: ",predictions.shape)
segmentation, prob = segment(img_ai,seed_points)
viewer.add_labels(segmentation,name="segmentation")

print(dice_coefficient(segmentation,ground_truth))


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





napari.run()

