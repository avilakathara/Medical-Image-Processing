from pathlib import Path
from skimage.segmentation import random_walker
from skimage.segmentation import flood
# from skimage.segmentation import expand_labels
# from skimage.morphology import ball
from skimage.morphology import binary_dilation
# from skimage.morphology import binary_erosion
# from skimage.morphology import binary_opening

# from skimage.morphology import binary_closing
# from skimage.morphology import area_closing
# from skimage.filters import gaussian
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2 as cv
from scipy.ndimage import affine_transform
from datetime import datetime
now = datetime.now()

# FOR THE REST OF THE GROUP:
# conda install -c anaconda pyamg
# pip install chardet


import pyamg

def convert_to_labels(drawn_contours):
    drawn_contours = drawn_contours.astype(int)

    for index,slice in enumerate(drawn_contours):
        # is this cheating?
        if index == 0 or index == len(drawn_contours)-1:
            drawn_contours[index] = np.ones(drawn_contours[0].shape)*2
        else:
            drawn_contours[index] = convert_to_labels2d(slice)
    return drawn_contours


def convert_to_labels2d(slice):
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

        dilation = binary_dilation(draw_contour,footprint=np.ones((3, 3)))
        filled_background = np.ones(slice.shape)
        filled_background[flood(dilation,background_start)] = 2

        dilation = binary_dilation(draw_contour,footprint=np.ones((2, 2)))
        img = np.ones(slice.shape)
        img[flood(dilation,background_start)] = 2
        img[dilation] = 0

        if np.count_nonzero(img==1) != 0:
            new_image[filled_background==1] = 0
            new_image[img==1] = 1
            continue

        dilation = binary_dilation(draw_contour,footprint=np.ones((1, 1)))
        img = np.ones(slice.shape)
        img[flood(dilation,background_start)] = 2
        img[dilation] = 0

        if np.count_nonzero(img==1) != 0:
            new_image[filled_background==1] = 0
            new_image[img==1] = 1
            continue

        new_image[draw_contour==1] = 1


    return new_image

    # return drawn_contours


def segment(image, seed_points):
    seed_points = seed_points.astype(int)
    start_time = time.time()
    prob = random_walker(image, seed_points.astype(int), beta=0.1, mode='cg_j',tol=0.001, copy=False, return_full_prob=True)
    print(time.time() - start_time, "seconds")

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




