import os
import napari
import json
import nrrd
import tkinter as tk
from tkinter import filedialog
from qtpy.QtWidgets import QMessageBox

from segmentation.segmentation import segment
from slice_select.optimization import get_optimal_slice
from uncertainty.uncertainty import calculate_uncertainty_fields

from process_patients import *
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
file_path = filedialog.askopenfilename()
window.destroy()

# INITIATE NAPARI
viewer = napari.Viewer()

# LOAD IMAGE
try:
    # TODO: convert ndarray into numpy array
    img, _ = nrrd.read(file_path)
except:
    img = np.load(file_path)
viewer.add_image(img, name="CT_SCAN", colormap="gray", interpolation2d="bicubic")

data = {'chosen_point': 0, 'chosen_normal': [0, 0, 0], 'chosen_axis': 'x', 'user_input': []}
with open('data.json', 'w') as f:
    json.dump(data, f)

segmentation = None
probabilities = None
uncertainty_field = None

# PRESS 'S' TO SEGMENT
@viewer.bind_key('s')
def get_segmentation(viewer):
    global segmentation
    global probabilities
    segmentation, probabilities = segment(img) # test
    viewer.add_image(segmentation, name="segmentation", colormap="gray", interpolation2d="bicubic")
    # try:
    #     segmentation = np.load("segmentation.npy")
    #     probabilities = np.load("probabilities.npy")
    #     viewer.add_labels(segmentation, name="segmentation")
    #     # viewer.add_image(probabilities, name="prob", colormap="gray", interpolation2d="bicubic")
    # except:
    #     segmentation, probabilities = segment(img)

# PRESS 'U' TO GET UNCERTAINTY FIELD
@viewer.bind_key('u')
def get_uncertainty_field(viewer):
    global uncertainty_field
    try:
        uncertainty_field = np.load("uncertainty.npy")
    except:
        uncertainty_field = calculate_uncertainty_fields(img, segmentation, probabilities)

    viewer.add_image(uncertainty_field, name="uncertainty_{}".format("u"), colormap="gray", interpolation2d="bicubic")

# count = 1

@viewer.bind_key('p')
def user_check(viewer):
    # Find optimal slice
    uncertainty, point, normal, chosen_axis = get_optimal_slice(uncertainty_field)

    # load the slice as a special image with a name n
    # Get the image
    img = viewer.layers['CT_SCAN'].data
    # rotate here is needed
    # TODO: this normal needs to be used to rotate the segmentation and the CT scan before taking the array at the point based on chosen axis
    chosen_slice = img[point]

    img_layer = viewer.add_image(chosen_slice, name="chosen_slice", colormap="gray", interpolation2d="bicubic")

    # Save the chosen values here into a JSON for future use if needed
    with open('data.json', 'r') as f:
        loaded_data = json.load(f)
    # Save info into json
    loaded_data["chosen_point"] = point
    loaded_data["chosen_normal"] = normal
    loaded_data["chosen_axis"] = chosen_axis
    with open('data.json', 'w') as f:
        json.dump(loaded_data, f)

@viewer.bind_key('c')
def user_input(viewer):
    # Get the annotations the user makes
    layer = viewer.layers['Points'].data
    user_input = []
    for val in layer:
        user_input.append(val[1:].tolist())

    # Save these annotations

    with open('data.json', 'r') as f:
        loaded_data = json.load(f)

    loaded_data["user_input"] = user_input

    with open('data.json', 'w') as f:
        json.dump(loaded_data, f)

    # Remove the points and chosen layer for next iteration
    # The two for loops are inefficient, but this needs to be done this way otherwise it bugs
    for layer in viewer.layers:
        if (layer.name == 'chosen_slice'):
            viewer.layers.remove(layer)
    for layer in viewer.layers:
        if (layer.name == "Points"):
            viewer.layers.remove(layer)

    # TODO: The modified slice has been gotten, we do something with this
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

# start the napari event loop
napari.run()

