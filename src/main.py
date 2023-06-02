import os
import napari
import json
import nrrd
import tkinter as tk
from tkinter import filedialog
from qtpy.QtWidgets import QMessageBox
import cv2 as cv

from segmentation.segmentation import *
from slice_select.optimization import get_optimal_slice
from slice_select.discreet_optimization import discreet_get_optimal_slice
from slice_select.rotation_methods import rotate, image_rotate_1, image_rotate_back_1, true_img_rot, true_img_rot_back
from uncertainty.evaluate_uncertainty import evaluate_uncertainty
from uncertainty.uncertainty import calculate_uncertainty_fields

from pathlib import Path

# PROMPT TO SELECT FILE
window = tk.Tk()
window.withdraw()
folder_path = filedialog.askdirectory()

window.destroy()

# INITIATE NAPARI
viewer = napari.Viewer()

# LOAD IMAGES
img = np.load(folder_path + str("/") + os.listdir(folder_path)[0])
structures = np.load(folder_path + str("/") + os.listdir(folder_path)[1]).astype(int)
ground_truth = np.zeros(structures.shape).astype(int)

# CHOOSE ORGAN
# 1=BrainStem,2=Chiasm,3=Mandible,4=OpticNerve_L,5=OpticNerve_R,6=Parotid_L,7=Parotid_R,8=Submandibular_L,9=Submandibular_R)
organ_choice = 3
ground_truth[structures == organ_choice] = 1

# USE BOUNDING BOX
z = np.any(ground_truth, axis=(1, 2))
y = np.any(ground_truth, axis=(0, 2))
x = np.any(ground_truth, axis=(0, 1))
zmin, zmax = np.where(z)[0][[0, -1]]
ymin, ymax = np.where(y)[0][[0, -1]]
xmin, xmax = np.where(x)[0][[0, -1]]
z_offset = 1
xy_offset = 10
zmin = max(0, zmin - z_offset)
ymin = max(0, ymin - xy_offset)
xmin = max(0, xmin - xy_offset)
zmax = min(len(img), zmax + z_offset + 1)
ymax = min(len(img[0]), ymax + xy_offset + 1)
xmax = min(len(img[0][0]), xmax + xy_offset + 1)

img = img[zmin:zmax, ymin:ymax, xmin:xmax]
ground_truth = ground_truth[zmin:zmax, ymin:ymax, xmin:xmax]

# SHOW IMAGES
viewer.add_image(img, name="CT_SCAN", colormap="gray", interpolation2d="bicubic")
# viewer.add_labels(ground_truth, name="GROUND TRUTH")

segmentation = None
probabilities = None
uncertainty_field = None
contours = None
seed_points = None
lw_layer = None
seg_layer = None
fetched_plane_index = None
chosen_slice = None
iterations = 0
normal = None
point = None
axis = None
chosen_layer = None


def create_contours(viewer):
    global seed_points
    global contours
    global lw_layer
    global mypath  # test shit
    global simulate_user_input
    global ground_truth
    global iterations
    global fetched_plane_index
    global axis
    global normal

    if iterations > 0:
        # rotated_ground_truth = image_rotate_1(ground_truth, normal)
        # if axis == "d1":
        #     rotated_ground_truth = rotate(ground_truth, [0, 0, 1], 315)
        #     viewer.add_image(rotated_ground_truth, name="rotated GT", colormap="gray", interpolation2d="bicubic")
        # elif axis == "x":
        #     rotated_ground_truth = ground_truth
        # else:
        #     rotated_ground_truth = true_img_rot(ground_truth, normal)
        #     viewer.add_labels(rotated_ground_truth.astype(int), name="rotated GT")

        # print(rotated_ground_truth.shape)
        if axis == "x":
            contours = create_contour(ground_truth[point])
        elif axis == "y":
            contours = create_contour(ground_truth[:, point, :])
        elif axis == "z":
            contours = create_contour(ground_truth[:, :, point])
        else:
            contours = create_contour(ground_truth[point])
        contours_display = np.full(ground_truth.shape, False)
        if axis == "x":
            contours_display[point] = contours
        elif axis == "y":
            contours_display[:, point, :] = contours
        elif axis == "z":
            contours_display[:, :, point] = contours
        else:
            contours_display[point] = contours
        # np.save("contours", contours)
        lw_layer = viewer.add_labels(contours_display, name='INPUT {}'.format(iterations), opacity=1.0)
    else:
        contours = automatic_contours(ground_truth)
        # np.save("contours", contours)
        lw_layer = viewer.add_labels(contours, name='INPUT {}'.format(iterations), opacity=1.0)


def get_segmentation(viewer):
    global segmentation
    global probabilities
    global seed_points
    global mypath
    global iterations
    global contours
    global normal
    global seg_layer

    if seed_points is None:
        seed_points = convert_to_labels(contours)
    else:
        new_labeled_slice = convert_to_labels2d(contours)
        if axis == "d1":
            # rotated_ground_truth = rotate(ground_truth, [0, 0, 1], 315)
            rotate_seed_points = rotate(seed_points, normal, 45)
            rotate_seed_points[point] = new_labeled_slice
            seed_points = rotate(rotate_seed_points, normal, 315)
            viewer.add_labels(seed_points, name="seedpoints debug".format(iterations))
        elif axis == "x":
            print(seed_points.shape)
            seed_points[point] = new_labeled_slice
        elif axis == "y":
            seed_points[:, point, :] = new_labeled_slice
        elif axis == "z":
            seed_points[:, :, point] = new_labeled_slice
        else:
            rotate_seed_points = true_img_rot(seed_points, normal)
            rotate_seed_points[point] = new_labeled_slice
            seed_points = true_img_rot_back(rotate_seed_points, normal)

    segmentation, probabilities = segment(img, seed_points)
    seg_layer = viewer.add_labels(segmentation, name="Segmentation {}".format(iterations))


def get_uncertainty_field(viewer, draw=False):
    global uncertainty_field
    uncertainty_field = calculate_uncertainty_fields(img, segmentation, probabilities)

    if draw:
        viewer.add_image(uncertainty_field, name="uncertainty_{}".format("u"), colormap="gray",
                         interpolation2d="bicubic")


def user_check(viewer, discrete=True):
    global fetched_plane_index
    global iterations
    global chosen_slice
    global normal
    global point
    global axis
    global chosen_layer

    # Find optimal slice
    # uncertainty, point, normal, chosen_axis = get_optimal_slice(uncertainty_field)
    uncertainty, point, normal, chosen_axis = discreet_get_optimal_slice(uncertainty_field, True, True, True)
    axis = chosen_axis

    print("Iteration {} - MAX UNCERTAINTY at plane {} = {}".format(iterations, chosen_axis, point))

    if chosen_axis == 'x' and discrete:
        chosen_slice = img[point]
    elif chosen_axis == 'y' and discrete:
        chosen_slice = img[:, point, :]
    elif chosen_axis == 'z' and discrete:
        chosen_slice = img[:, :, point]
    elif chosen_axis == "d1" and discrete:
        image_unrotate = true_img_rot(img, [0, 0, 1])
        chosen_slice = image_unrotate[point]

    chosen_layer = viewer.add_image(chosen_slice, name="chosen_slice", colormap="gray", interpolation2d="bicubic")


def confirm_dialog(title, message):
    msg_box = QMessageBox()
    msg_box.setWindowTitle(title)
    msg_box.setText(message)
    msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    msg_box.setDefaultButton(QMessageBox.No)
    return msg_box.exec_() == QMessageBox.Yes


# MAKE ANNOTATIONS
@viewer.bind_key('a')
def on_press_a(viewer):
    try:
        viewer.layers.remove(lw_layer)
    except:
        pass
    create_contours(viewer)


@viewer.bind_key('z')
def test(viewer):
    global img
    roti = true_img_rot(img, [0, 0, 1])
    viewer.add_image(roti, name="rotated 45", colormap="gray", interpolation2d="bicubic")
    roti = true_img_rot_back(roti, [0, 0, 1])
    viewer.add_image(roti, name="rotated -45", colormap="gray", interpolation2d="bicubic")


# MAKE SEGMENTATION
@viewer.bind_key('s')
def on_press_s(viewer):
    global iterations
    global ground_truth
    global segmentation
    global uncertainty_field
    global lw_layer
    global seg_layer
    global chosen_layer

    try:
        viewer.layers.remove(lw_layer)
    except:
        pass
    try:
        viewer.layers.remove(seg_layer)
    except:
        pass
    try:
        viewer.layer.remove(chosen_layer)
    except:
        pass

    iterations += 1
    get_segmentation(viewer)
    get_uncertainty_field(viewer)
    # print(evaluate_uncertainty(ground_truth, segmentation, uncertainty_field))
    user_check(viewer)


# TODO: uncomment this code and use it to automate main, and delete key bindings
# user_interaction = 0
# diff_score = 1.0
# threshold = 0.005
# while(diff_score > threshold):
#     user_interaction += 1
#     create_contours()
#     iterations += 1
#     get_segmentation()
#     get_uncertainty_field()
#     user_check()
#
#     diff_score = np.sum(abs(segmentation-ground_truth))/np.product(segmentation.shape)
#     print(diff_score)
#
#     # TODO: make sure slice is not already annotated, this could happen when the threshold is never reached
#
#
# viewer.add_labels(segmentation.astype(int),name="final segmentation")
# viewer.add_labels(ground_truth.astype(int),name="ground truth")
#
# print(user_interaction)

# INITIATE
napari.run()
