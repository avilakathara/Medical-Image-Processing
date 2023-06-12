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
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.segmentation import random_walker


def dice_coefficient(mask1, mask2):
    intersection = np.sum(mask1 * mask2)
    total = np.sum(mask1) + np.sum(mask2)
    dice = (2.0 * intersection) / total
    return dice

def false_positives(segmentation,ground_truth):
    return np.sum((segmentation == 1) & (ground_truth == 0))

def false_negatives(segmentation,ground_truth):
    return np.sum((segmentation == 0) & (ground_truth == 1))

def synthetic_data():
    image = np.zeros((100,100))
    image = cv.circle(image, (20,50), 10, 1, -1)
    image = cv.circle(image,(80,50),10,1,-1)
    image[49:51,20:80] = 1

    seed_points = np.zeros(image.shape)
    seed_points[:,0] = 2
    seed_points[:,99] = 2
    seed_points[49:51,20:80] = 1

    segmentation1,prob = segment(image,seed_points)
    segmentation2,prob = segment(image,seed_points,pred=np.array([image]))
    plt.imshow(segmentation1,cmap='gray')
    plt.show()
    plt.imshow(segmentation2,cmap='gray')
    plt.show()

# window = tk.Tk()
# window.withdraw()
# folder_path = filedialog.askdirectory()
# window.destroy()
# viewer = napari.Viewer()
# # 1=BrainStem,2=Chiasm,3=Mandible,4=OpticNerve_L,5=OpticNerve_R,6=Parotid_L,7=Parotid_R,8=Submandibular_L,9=Submandibular_R)
# chosen_organ = 3
# img = np.load(folder_path + str("/") + os.listdir(folder_path)[1])
# global predictions
# predictions = []
# for i in range(2,8):
#     prediction = np.load(folder_path + str("/") + os.listdir(folder_path)[i])
#     prediction[prediction != chosen_organ] = 0
#     prediction[prediction != 0] = 1
#     predictions.append(prediction.astype(int))
# predictions = np.array(predictions)
# ground_truth = np.load(folder_path + str("/") + os.listdir(folder_path)[9])
# ground_truth[ground_truth != chosen_organ] = 0
# ground_truth[ground_truth != 0] = 1
# ground_truth = ground_truth.astype(int)
# seed_points = convert_to_labels(automatic_contours(ground_truth))
# segmentation, prob = segment(img,seed_points,pred=predictions)
# viewer.add_image(img, name="CT_SCAN", colormap="gray", interpolation2d="bicubic")
# viewer.add_labels(segmentation.astype(int),name='segmentation')


# approach = 4
#
# # APPROACH 1
# # majority vote
# if approach == 1:
#     maj_vote = np.around(np.sum(predictions,axis=0)/len(predictions)).astype(int)
#     seed_points = np.zeros(maj_vote.shape)
#     for index,slice in enumerate(maj_vote):
#         dilated = binary_dilation(slice,footprint=np.ones((2,2)))
#         eroded = binary_erosion(slice,footprint=np.ones((2,2)))
#         indices = np.argwhere(dilated == 0)[0]
#         seed_points[index][flood(dilated,(indices[0],indices[1]))] = 2
#         seed_points[index][eroded == 1] = 1
#
#     # contours = []
#     # for slice in maj_vote:
#     #     contours.append(create_contour(slice))
#     # contours = np.array(contours).astype(int)
#     # seed_points = convert_to_labels(contours)
#
# # APPROACH 2
# # unanimous votes
# if approach == 2:
#     sum_pred = np.sum(predictions,axis=0).astype(int)
#     seed_points = np.zeros(sum_pred.shape)
#     seed_points[sum_pred == 0] = 2
#     seed_points[sum_pred == 6] = 1
#
# # APPROACH 3
# # selected positive unanimous votes
# if approach == 3:
#     sum_pred = np.sum(predictions,axis=0).astype(int)
#     seed_points = np.zeros(sum_pred.shape)
#     thres = 0.9
#     for index,slice in enumerate(sum_pred):
#         if np.count_nonzero(slice) == 0:
#             continue
#         if np.count_nonzero(slice==6)/np.count_nonzero(slice) > thres:
#             seed_points[index][slice==0] = 2
#             seed_points[index][slice==6] = 1
#
#
# # APPROACH 4
# # use baseline seeds
# if approach == 4:
#     seed_points = convert_to_labels(automatic_contours(ground_truth))
#
# # viewer.add_image(img, name="CT_SCAN", colormap="gray", interpolation2d="bicubic")
# viewer.add_image(img_ai, name="CT_SCAN AI", colormap="gray", interpolation2d="bicubic")
# viewer.add_labels(seed_points.astype(int), name="seed points")
#
#
# print("img shape: ",img_ai.shape,", seed_points shape: ",seed_points.shape,", predictions shape: ",predictions.shape)
# segmentation, prob = segment(img_ai,seed_points)
# viewer.add_labels(segmentation,name="segmentation")
#
# print(dice_coefficient(segmentation,ground_truth))
def evaluate(chosen_organ):
    path = Path(r'C:\Users\Bram\Documents\RP\miccai_data_numpy\part4')
    results = []
    for p in path.rglob("img.npy"):

        patient_base_path = p.parent
        patient_id = patient_base_path.stem
        img = np.load(patient_base_path.joinpath("img_ai.npy"))
        ground_truth = np.load(patient_base_path.joinpath("structures_ai.npy")).astype(int)
        predictions = []
        for i in range(1,7):
            prediction = np.load(patient_base_path.joinpath("maskpred_"+str(i)+".npy"))
            prediction[prediction != chosen_organ] = 0
            prediction[prediction != 0] = 1
            predictions.append(prediction.astype(int))
        predictions = np.array(predictions)
        ground_truth[ground_truth != chosen_organ] = 0
        ground_truth[ground_truth != 0] = 1
        ground_truth = ground_truth.astype(int)

        seed_points = convert_to_labels(automatic_contours(ground_truth))
        segmentation1, prob1 = segment(img,seed_points)
        segmentation2, prob2 = segment(img,seed_points,pred=predictions)
        print(str(patient_id)+"s results for organ "+str(chosen_organ)+": ")
        d1 = dice_coefficient(segmentation1,ground_truth)
        d2 = dice_coefficient(segmentation2,ground_truth)
        if d1 > d2:
            print("it made it worse!!")
        fp1 = false_positives(segmentation1,ground_truth)
        fp2 = false_positives(segmentation2,ground_truth)
        fn1 = false_negatives(segmentation1,ground_truth)
        fn2 = false_negatives(segmentation2,ground_truth)
        results.append((d1,d2,fp1,fp2,fn1,fn2,str(patient_id)))
        print(results[-1])
    return results


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


print("weights are 0.5 and 0.5")
all_results = []
for organ in range(1,10):
    results = evaluate(organ)
    all_results.append(results)
file_path = '../data.txt'
with open(file_path, 'w') as file:
    file.write(str(all_results))
# napari.run()

# 0522c0555s results for organ 1:
#     (0.042282694093279054, 0.06153405838580424, 203, 44, 13206, 13072, '0522c0555')
# 0522c0576s results for organ 1:
#     (0.09229600557067798, 0.09823434991974318, 313, 55, 14026, 13990, '0522c0576')
# 0522c0598s results for organ 1:
#     it made it worse!!
# (0.0410568836500894, 0.018319989143710136, 190, 2, 14291, 14466, '0522c0598')
# 0522c0659s results for organ 1:
#     (0.033646840641144524, 0.07120925066962827, 124, 42, 14466, 14175, '0522c0659')
# 0522c0661s results for organ 1:
#     (0.022052337547780066, 0.09389937999905343, 72, 28, 19884, 19117, '0522c0661')
# 0522c0667s results for organ 1:
#     (0.0451819829870311, 0.0969655172413793, 237, 16, 13457, 13078, '0522c0667')
# 0522c0669s results for organ 1:
#     (0.049928406897839755, 0.07418894609515539, 255, 116, 15006, 14809, '0522c0669')
# 0522c0708s results for organ 1:
#     (0.059569146819757855, 0.08063974946873952, 19, 111, 16526, 16329, '0522c0708')
# 0522c0727s results for organ 1:
#     (0.0587792012057272, 0.08343163538873995, 299, 139, 17187, 16955, '0522c0727')
# 0522c0746s results for organ 1:
#     it made it worse!!
# (0.029007886174378206, 0.014455172413793103, 53, 177, 17554, 17686, '0522c0746')
# 0522c0555s results for organ 2:
#     (0.3969924812030075, 0.5956416464891041, 310, 67, 91, 100, '0522c0555')
# 0522c0576s results for organ 2:
#     (0.392831541218638, 0.7526555386949925, 738, 28, 109, 135, '0522c0576')
# 0522c0598s results for organ 2:
#     (0.5192660550458715, 0.5279187817258884, 236, 9, 288, 363, '0522c0598')
# 0522c0659s results for organ 2:
#     (0.32869565217391306, 0.6277665995975855, 651, 31, 121, 154, '0522c0659')

