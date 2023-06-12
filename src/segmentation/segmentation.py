from pathlib import Path

import numpy as np
import time
import math
import cv2 as cv
import skimage.segmentation.random_walker_segmentation as rw
# from skimage.segmentation import random_walker
from skimage.segmentation import flood
from skimage.morphology import binary_dilation
from scipy.ndimage import affine_transform
from datetime import datetime

now = datetime.now()

# path = r'C:\Users\Bram\Documents\RP\miccai_data_numpy\part1\0522c0002'
# img = np.load(Path(path + str("\img.npy")))
# structures = np.load(Path(path + str("\structures.npy")))
# z = np.any(structures, axis=(1, 2))
# y = np.any(structures, axis=(0, 2))
# x = np.any(structures, axis=(0, 1))
# zmin, zmax = np.where(z)[0][[0, -1]]
# ymin, ymax = np.where(y)[0][[0, -1]]
# xmin, xmax = np.where(x)[0][[0, -1]]
# z_offset = 1
# xy_offset = 10
# zmin = max(0, zmin - z_offset)
# ymin = max(0, ymin - xy_offset)
# xmin = max(0, xmin - xy_offset)
# zmax = min(len(img), zmax + z_offset + 1)
# ymax = min(len(img[0]), ymax + xy_offset + 1)
# xmax = min(len(img[0][0]), xmax + xy_offset + 1)
#
# img = img[zmin:zmax, ymin:ymax, xmin:xmax]
# structures = structures[zmin:zmax, ymin:ymax, xmin:xmax]
# ground_truth = np.zeros(structures.shape).astype(int)
#
# # CHOOSE ORGAN
# # 1=BrainStem,2=Chiasm,3=Mandible,4=OpticNerve_L,5=OpticNerve_R,6=Parotid_L,7=Parotid_R,8=Submandibular_L,9=Submandibular_R)
# organ_choice = 3
# ground_truth[structures == organ_choice] = 1


predictions = None


def convert_to_labels(drawn_contours, z_bound_down=-1, z_bound_up=-1):
    drawn_contours = drawn_contours.astype(int)
    for index,slice in enumerate(drawn_contours):
        if index == 0 or index == len(drawn_contours)-1:
            drawn_contours[index] = np.ones(drawn_contours[0].shape)*2
        else:
            drawn_contours[index] = convert_to_labels2d(slice)
    return drawn_contours


def convert_to_labels2d(slice,dil_size=3):
    # drawn_contours = drawn_contours.astype(int)
    #
    # for index,slice in enumerate(drawn_contours):
    #
    if np.count_nonzero(slice) == 0:
        return slice
    new_image = np.ones(slice.shape)*2
    background_start = (0,0)
    for i in range(len(slice)):
        if slice[background_start[0],background_start[1]] == 0:
            break
        for j in range(len(slice[0])):
            if slice[i,j] == 0:
                background_start = (i,j)
                break
    contours, hierarchy = cv.findContours(slice.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        draw_contour = np.zeros(slice.shape)
        for point in contour:
            draw_contour[point[0][1], point[0][0]] = 1
        dilation_possible = False
        for i in range(dil_size):
            dilation = binary_dilation(draw_contour,footprint=np.ones((dil_size-i, dil_size-i))).astype(int)
            dilation[flood(dilation,background_start)] = 2
            if np.count_nonzero(dilation) < len(dilation)*len(dilation[0]):
                # print("dilation 3")
                # a region inside contour remains, so this dilation size works
                # flip contour with region inside of it and put into the new image
                new_image[dilation == 1] = 0
                new_image[dilation == 0] = 1
                dilation_possible = True
                break

        # dilation = binary_dilation(draw_contour,footprint=np.ones((2,2))).astype(int)
        # dilation[flood(dilation,background_start)] = 2
        # if np.count_nonzero(dilation) < len(dilation)*len(dilation[0]):
        #     # print("dilation 2")
        #     # a region inside contour remains, so this dilation size works
        #     # flip contour with region inside of it and put into the new image
        #     new_image[dilation == 1] = 0
        #     new_image[dilation == 0] = 1
        #     continue
        #
        # dilation = binary_dilation(draw_contour,footprint=np.ones((1,1))).astype(int)
        # dilation[flood(dilation,background_start)] = 2
        # if np.count_nonzero(dilation) < len(dilation)*len(dilation[0]):
        #     # print("dilation 1")
        #     # a region inside contour remains, so this dilation size works
        #     # flip contour with region inside of it and put into the new image
        #     new_image[dilation == 1] = 0
        #     new_image[dilation == 0] = 1
        #     continue

        # if no dilation works, we simply label contour itself
        # print("no dilation, just take contour")
        if not dilation_possible:
            new_image[draw_contour == 1] = 1


    return new_image

    # return drawn_contours
def automatic_contours(ground_truth):
    result = np.zeros(ground_truth.shape,dtype=bool)
    target = int(len(ground_truth)/2)

    for i in range(target):
        if np.count_nonzero(ground_truth[target-i]) > 0:
            target = target - i
            break
        if np.count_nonzero(ground_truth[target+i]) > 0:
            target = target+i
            break

    result[target] = create_contour(ground_truth[target])
    return result

def create_contour(ground_truth):
    if np.count_nonzero(ground_truth) == 0:
        return ground_truth
    result = np.zeros(ground_truth.shape,dtype=bool)
    im = ground_truth.astype(np.uint8)
    contours, hierarchy = cv.findContours(im, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        for point in contour:
            result[point[0][1], point[0][0]] = True
    return result

def segment(image, seed_points, pred = None):
    global predictions
    if pred is not None:
        predictions = pred
        rw._compute_weights_3d = modified_weights

    seed_points = seed_points.astype(int)
    start_time = time.time()
    prob = rw.random_walker(image, seed_points.astype(int), beta=20, mode='cg_j',tol=0.1, copy=False, return_full_prob=True)
    # print(time.time() - start_time, "seconds")

    labels = np.zeros(image.shape).astype(int)
    labels[prob[0]>=0.5] = 1
    return labels, prob


def rot(image, axis, angle_deg):

    # Define the rotation axis and angle
    axis = np.array(axis)  # Example: Rotate around the X-axis
    #angle_deg = 45.0

    # Convert the angle to radians
    angle_rad = np.radians(angle_deg)

    # Calculate the rotation matrix
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    u = axis / np.linalg.norm(axis)
    rotation_matrix = np.array([
        [cos_theta + u[0] * u[0] * (1 - cos_theta), u[0] * u[1] * (1 - cos_theta) - u[2] * sin_theta,
         u[0] * u[2] * (1 - cos_theta) + u[1] * sin_theta],
        [u[1] * u[0] * (1 - cos_theta) + u[2] * sin_theta, cos_theta + u[1] * u[1] * (1 - cos_theta),
         u[1] * u[2] * (1 - cos_theta) - u[0] * sin_theta],
        [u[2] * u[0] * (1 - cos_theta) - u[1] * sin_theta, u[2] * u[1] * (1 - cos_theta) + u[0] * sin_theta,
         cos_theta + u[2] * u[2] * (1 - cos_theta)]
    ])

    # Define the center point for rotation
    center = np.array(image.shape) / 2.0
    translation = center - np.dot(rotation_matrix, center)
    # Perform the 3D rotation
    rotated_image = affine_transform(image, rotation_matrix, offset=translation, order=1, mode='constant', cval=-1024.0)
    #rotated_image = affine_transform(image, rotation_matrix, offset=translation, order=1, mode='nearest', cval=0.0)
    return rotated_image

# Image: array of points
def undo_rot(image, axis, angle_deg):

    # Define the rotation axis and angle
    axis = np.array(axis)  # Example: Rotate around the X-axis
    #angle_deg = 45.0


    # Convert the angle to radians
    angle_rad = np.radians(360 - angle_deg)

    # Calculate the rotation matrix
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    u = axis / np.linalg.norm(axis)
    rotation_matrix = np.array([
        [cos_theta + u[0] * u[0] * (1 - cos_theta), u[0] * u[1] * (1 - cos_theta) - u[2] * sin_theta,
         u[0] * u[2] * (1 - cos_theta) + u[1] * sin_theta],
        [u[1] * u[0] * (1 - cos_theta) + u[2] * sin_theta, cos_theta + u[1] * u[1] * (1 - cos_theta),
         u[1] * u[2] * (1 - cos_theta) - u[0] * sin_theta],
        [u[2] * u[0] * (1 - cos_theta) - u[1] * sin_theta, u[2] * u[1] * (1 - cos_theta) + u[0] * sin_theta,
         cos_theta + u[2] * u[2] * (1 - cos_theta)]
    ])
    #r_m = np.linalg.inv(rotation_matrix)
    # Define the center point for rotation
    center = np.array(image.shape) / 2.0
    translation = center - np.dot(rotation_matrix, center)
    # Perform the 3D rotation
    rotated_image = affine_transform(image, rotation_matrix, offset=translation, order=1, mode='constant', cval=-1024.0)
    #rotated_image = affine_transform(image, rotation_matrix, offset=translation, order=1, mode= 'nearest', cval=0.0)

    return rotated_image

def angle_between_vectors(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a**2 for a in vec1))
    magnitude2 = math.sqrt(sum(b**2 for b in vec2))
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    angle_rad = math.acos(cosine_angle)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def image_rotate(image, normal):
    axis = np.array([1,0,0])
    angle = angle_between_vectors(axis, normal)
    if angle > 5:
        rotated_image = rot(image, axis, angle)
    # axis = np.array([0, 1, 0])
    # angle = angle_between_vectors(axis, normal)
    # if angle > 5:
    #     rotated_image = rot(rotated_image, axis, angle)
    # axis = np.array([0, 0, 1])
    # angle = angle_between_vectors(axis, normal)
    # if angle > 5:
    #     rotated_image = rot(rotated_image, axis, angle)

    return rotated_image

def image_rotate_back(image, normal):
    axis = np.array([1,0,0])
    angle = angle_between_vectors(axis, normal)
    if angle > 5:
        rotated_image = undo_rot(image, axis, angle)
    # axis = np.array([0, 1, 0])
    # angle = angle_between_vectors(axis, normal)
    # if angle > 5:
    #     rotated_image = rot(rotated_image, axis, angle)
    # axis = np.array([0, 0, 1])
    # angle = angle_between_vectors(axis, normal)
    # if angle > 5:
    #     rotated_image = rot(rotated_image, axis, angle)

    return rotated_image
def prediction_weights(predictions,data, spacing, beta, eps, multichannel):
    sum_preds = np.sum(predictions,axis=0)
    data = np.zeros(data.shape)
    for i in range(len(sum_preds)):
        for j in range(len(sum_preds[0])):
            for k in range(len(sum_preds[0][0])):
                data[i,j,k,0] = sum_preds[i,j,k]

    # Weight calculation is main difference in multispectral version
    # Original gradient**2 replaced with sum of gradients ** 2
    gradients = np.concatenate(
        [np.diff(data[..., 0], axis=ax).ravel() / spacing[ax]
         for ax in [2, 1, 0] if data.shape[ax] > 1], axis=0) ** 2

    for channel in range(1, data.shape[-1]):
        gradients += np.concatenate(
            [np.diff(data[..., channel], axis=ax).ravel() / spacing[ax]
             for ax in [2, 1, 0] if data.shape[ax] > 1], axis=0) ** 2

    # All channels considered together in this standard deviation
    scale_factor = -beta / (10 * data.std())
    if multichannel:
        # New final term in beta to give == results in trivial case where
        # multiple identical spectra are passed.
        scale_factor /= np.sqrt(data.shape[-1])
    weights = np.exp(scale_factor * gradients)
    weights += eps

    weights = -weights
    return weights

def modified_weights(data, spacing, beta, eps, multichannel):
    global predictions
    sum_preds = np.sum(predictions,axis=0)
    data = np.zeros(data.shape)
    for i in range(len(sum_preds)):
        for j in range(len(sum_preds[0])):
            for k in range(len(sum_preds[0][0])):
                data[i,j,k,0] = sum_preds[i,j,k]

    # Weight calculation is main difference in multispectral version
    # Original gradient**2 replaced with sum of gradients ** 2
    gradients = np.concatenate(
        [np.diff(data[..., 0], axis=ax).ravel() / spacing[ax]
         for ax in [2, 1, 0] if data.shape[ax] > 1], axis=0) ** 2

    for channel in range(1, data.shape[-1]):
        gradients += np.concatenate(
            [np.diff(data[..., channel], axis=ax).ravel() / spacing[ax]
             for ax in [2, 1, 0] if data.shape[ax] > 1], axis=0) ** 2

    # All channels considered together in this standard deviation
    scale_factor = -beta / (10 * data.std())
    if multichannel:
        # New final term in beta to give == results in trivial case where
        # multiple identical spectra are passed.
        scale_factor /= np.sqrt(data.shape[-1])
    weights = np.exp(scale_factor * gradients)
    weights += eps

    weights = -weights
    pred_weights = prediction_weights(predictions,data, spacing, beta, eps, multichannel)
    w_1 = 0.5
    w_2 = 0.5
    return w_1*weights + w_2*pred_weights


