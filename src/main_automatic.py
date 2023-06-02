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
from slice_select.rotation_methods import rotate, image_rotate_1, image_rotate_back_1
from uncertainty.uncertainty import calculate_uncertainty_fields

from pathlib import Path


def confirm_dialog(title, message):
    msg_box = QMessageBox()
    msg_box.setWindowTitle(title)
    msg_box.setText(message)
    msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    msg_box.setDefaultButton(QMessageBox.No)
    return msg_box.exec_() == QMessageBox.Yes


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



data = {'chosen_point': 0, 'chosen_normal': [0, 0, 0], 'chosen_axis': 'x', 'user_input': []}
with open('data.json', 'w') as f:
    json.dump(data, f)

# DECIDE WHETHER OR NOT TO USE REAL USER INPUT
simulate_user_input = True

segmentation = None
probabilities = None
uncertainty_field = None
click_count = 0
contours = None
seed_points = None
lw_layer = None
fetched_plane_index = None
chosen_slice = None
iterations = 0
normal = None
point = None
axis = None

def create_contours():
    global seed_points
    global contours
    global ground_truth
    global iterations
    global fetched_plane_index
    global axis

    if iterations > 0:
        #rotated_ground_truth = image_rotate_1(ground_truth, normal)
        if axis == "d1":
            rotated_ground_truth = rotate(ground_truth, [0, 0, 1], 315)
            # viewer.add_image(rotated_ground_truth, name="rotated GT", colormap="gray", interpolation2d="bicubic")
        else:
            rotated_ground_truth = ground_truth

        contours = auto_add_contours(rotated_ground_truth[point])
        # np.save("contours", contours)
        # lw_layer = viewer.add_labels(contours, name='additional contours (automatic)', opacity=1.0)
    else:
        contours = automatic_contours(ground_truth)
        # np.save("contours", contours)
        # lw_layer = viewer.add_labels(contours, name='contours (automatic)', opacity=1.0)


def get_segmentation():
    global segmentation
    global probabilities
    global seed_points
    global iterations
    global contours
    global normal

    if seed_points is None:
        seed_points = convert_to_labels(contours)
    else:
        new_labeled_slice = convert_to_labels2d(contours)
        if axis == "d1":
            #rotated_ground_truth = rotate(ground_truth, [0, 0, 1], 315)
            rotate_seed_points = rotate(seed_points, normal, 45)
            rotate_seed_points[point] = new_labeled_slice
            seed_points = rotate(rotate_seed_points, normal, 315)
            # viewer.add_labels(seed_points, name="seedpoints debug".format(iterations))
        elif axis == "x":
            seed_points[point] = new_labeled_slice
        else:
            rotate_seed_points = image_rotate_1(seed_points,normal)
            rotate_seed_points[point] = new_labeled_slice
            seed_points = image_rotate_back_1(rotate_seed_points,normal)

    segmentation, probabilities = segment(img, seed_points)


def get_uncertainty_field(draw=False):
    global uncertainty_field
    uncertainty_field = calculate_uncertainty_fields(img, segmentation, probabilities)

    # if draw:
    #     viewer.add_image(uncertainty_field, name="uncertainty_{}".format("u"), colormap="gray",
    #                      interpolation2d="bicubic")


def user_check():
    global fetched_plane_index
    global iterations
    global chosen_slice
    global normal
    global point
    global axis
    # Find optimal slice
    # uncertainty, point, normal, chosen_axis = get_optimal_slice(uncertainty_field)
    uncertainty, point, normal, chosen_axis = discreet_get_optimal_slice(uncertainty_field, True, False, False)
    axis = chosen_axis

    print("Iteration {} - MAX UNCERTAINTY at plane z = {}".format(iterations, point))
    print(point,normal,chosen_axis)
    # load the slice as a special image with a name n
    # Get the image
    # img = viewer.layers['CT_SCAN'].data
    # rotate here is needed
    # # TODO: this normal needs to be used to rotate the segmentation and the CT scan before taking the array at the point based on chosen axis
    discreet = True
    if chosen_axis == 'x' and discreet:
        chosen_slice = img[point]
    elif chosen_axis == 'y' and discreet:
        chosen_slice = img[:, point, :]
    elif chosen_axis == 'z' and discreet:
        chosen_slice = img[:, :, point]
    elif chosen_axis == "d1" and discreet:
        image_unrotate = rotate(img, [0, 0, 1], 315)
        chosen_slice = image_unrotate[point]

    # viewer.add_image(chosen_slice, name="chosen_slice", colormap="gray", interpolation2d="bicubic")

    # Save the chosen values here into a JSON for future use if needed
    # with open('data.json', 'r') as f:
    #     loaded_data = json.load(f)
    # # Save info into json
    # loaded_data["chosen_point"] = point
    # loaded_data["chosen_normal"] = normal
    # loaded_data["chosen_axis"] = chosen_axis
    # with open('data.json', 'w') as f:
    #     json.dump(loaded_data, f)


# MAKE ANNOTATIONS
# @viewer.bind_key('a')
# def on_press_a(viewer):
#     try:
#         viewer.layers.remove(lw_layer)
#     except:
#         pass
#     create_contours(viewer)

user_interaction = 0
diff_score = 1.0
threshold = 0.005
while(diff_score > threshold):
    user_interaction += 1
    create_contours()
    iterations += 1
    get_segmentation()
    get_uncertainty_field()
    user_check()

    diff_score = np.sum(abs(segmentation-ground_truth))/np.product(segmentation.shape)
    print(diff_score)

    # TODO: make sure slice is not already annotated, this could happen when the threshold is never reached


viewer.add_labels(segmentation.astype(int),name="final segmentation")
viewer.add_labels(ground_truth.astype(int),name="ground truth")

print(user_interaction)

# @viewer.bind_key('z')
# def test(viewer):
#     global img
#     roti = rot(img, [0,0,1], 45)
#     viewer.add_image(roti, name="rotated 45", colormap="gray", interpolation2d="bicubic")
#     roti = rot(img, [0, 0, 1], 315)
#     viewer.add_image(roti, name="rotated -45", colormap="gray", interpolation2d="bicubic")
#
#
# # MAKE SEGMENTATION
# @viewer.bind_key('s')
# def on_press_s(viewer):
#     global iterations
#     iterations += 1
#     get_segmentation(viewer)
#     get_uncertainty_field(viewer)
#     user_check(viewer)
#
#     diff_score = np.sum(abs(segmentation-ground_truth))/np.product(segmentation.shape)
#     print(diff_score)
#
#
# @viewer.bind_key('c')
# def user_input(viewer):
#     # Get the annotations the user makes
#     layer = viewer.layers['Points'].data
#     user_input = []
#     for val in layer:
#         user_input.append(val[1:].tolist())
#
#     # Save these annotations
#
#     with open('data.json', 'r') as f:
#         loaded_data = json.load(f)
#
#     loaded_data["user_input"] = user_input
#
#     with open('data.json', 'w') as f:
#         json.dump(loaded_data, f)
#
#     # Remove the points and chosen layer for next iteration
#     # The two for loops are inefficient, but this needs to be done this way otherwise it bugs
#     for layer in viewer.layers:
#         if (layer.name == 'chosen_slice'):
#             viewer.layers.remove(layer)
#     for layer in viewer.layers:
#         if (layer.name == "Points"):
#             viewer.layers.remove(layer)
#

# INITIATE
napari.run()

#
# @viewer.bind_key('.')
# def test(viewer):
#
#     # the following is assuming we drew a line!
#
#     with open('data.txt', 'w') as f:
#         lines = viewer.layers['Shapes'].data
#         f.write(str(get_line_points(lines)))
#         lines = viewer.layers['Shapes [1]'].data
#         f.write('\n')
#         f.write(str(get_line_points(lines)))
#
# def get_line_points(lines):
#     all_points = []
#     for line in lines:
#         # add all points that lay on the line
#         print(line)
#
#         if line[0][1]-line[1][1] != 0:
#             a = (line[0][2]-line[1][2])/(line[0][1]-line[1][1])
#             b = line[0][2]-a*line[0][1]
#             for x in range(int(min(line[0][1],line[1][1])), int(max(line[0][1],line[1][1]))):
#                 all_points.append([int(line[0][0]),x,int(a*x + b)])
#         else:
#             for y in range(int(min(line[0][2],line[1][2])), int(min(line[0][2],line[1][2]))):
#                 all_points.append([int(line[0][0]),int(line[0][1]),y])
#     return all_points
