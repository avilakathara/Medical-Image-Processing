import os
import napari
import json
import nrrd
import tkinter as tk
from tkinter import filedialog
from qtpy.QtWidgets import QMessageBox
import cv2 as cv

from segmentation.livewire import *
from segmentation.segmentation import *
from slice_select.optimization import get_optimal_slice
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
z_offset = 2
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
livewire = None
click_count = 0
contours = None
seed_points = None
lw_layer = None
fetched_plane_index = None

iterations = 0


def create_contours(viewer):
    global livewire
    global click_count
    global seed_points
    global contours
    global lw_layer
    global mypath  # test shit
    global simulate_user_input
    global ground_truth
    global iterations
    global fetched_plane_index

    if livewire is not None:
        contours = livewire.drawing
        livewire = None
        # np.save("contours", contours)
        return

    if simulate_user_input:
        if iterations > 0:
            contours = auto_add_contours(contours, ground_truth, segmentation, fetched_plane_index)
            # np.save("contours", contours)
            lw_layer = viewer.add_labels(contours, name='additional contours (automatic)', opacity=1.0)
        else:
            contours = automatic_contours(ground_truth)
            # np.save("contours", contours)
            lw_layer = viewer.add_labels(contours, name='contours (automatic)', opacity=1.0)
    else:
        livewire = LiveWire2(img, sigma=5.0)
        lw_layer = viewer.add_labels(livewire.drawing,
                                     name='contours', opacity=1.0)

    def valid(coords):
        return 0 <= round(coords[0]) < img.shape[1] and 0 <= round(coords[1]) < img.shape[2]

    click_count = 0

    @lw_layer.mouse_move_callbacks.append
    def mouse_move(layer, event):
        global livewire
        global click_count
        if livewire is None:
            return
        if livewire.current_slice == -1:
            livewire.current_slice = round(event.position[0])
            livewire.laplacian = abs(cv.Laplacian(img[livewire.current_slice], cv.CV_32F))
            livewire.max_gradient = np.max(livewire.laplacian)
        if livewire.current_slice != round(event.position[0]):
            click_count = 0
            livewire.reset()
            livewire.current_slice = round(event.position[0])

            livewire.laplacian = abs(cv.Laplacian(img[livewire.current_slice], cv.CV_32F))
            livewire.max_gradient = np.max(livewire.laplacian)

        coords = event.position[1:]
        if valid(coords):
            indices_start = flat_to_indices(livewire.start, img.shape[2])
            dist_to_start = np.sqrt(((coords[0] - indices_start[0]) ** 2) + ((coords[1] - indices_start[1]) ** 2))
            if click_count > 2 and dist_to_start < 10:
                # close contour if close to start
                livewire.select(livewire.start)
            else:
                livewire.select(flatten_indices([round(coords[0]), round(coords[1])], img.shape[2]))
            layer.data = livewire.current_drawing + livewire.drawing

    @lw_layer.mouse_drag_callbacks.append
    def mouse_click(layer, event):
        global click_count
        global livewire
        if livewire is None:
            return
        livewire.clicked()

        click_count += 1


def get_segmentation(viewer):
    global segmentation
    global probabilities
    global seed_points
    global mypath
    global iterations
    if seed_points is None:
        seed_points = convert_to_labels(contours)
    segmentation, probabilities = segment(img, seed_points)
    viewer.add_labels(segmentation, name="Segmentation {}".format(iterations))


def get_uncertainty_field(viewer, draw=False):
    global uncertainty_field
    uncertainty_field = calculate_uncertainty_fields(img, segmentation, probabilities)

    if draw:
        viewer.add_image(uncertainty_field, name="uncertainty_{}".format("u"), colormap="gray",
                         interpolation2d="bicubic")


def user_check(viewer):
    global fetched_plane_index
    global iterations

    # Find optimal slice
    uncertainty, point, normal, chosen_axis = get_optimal_slice(uncertainty_field)

    print("Iteration {} - MAX UNCERTAINTY at plane z = {}".format(iterations, point))

    # load the slice as a special image with a name n
    # Get the image
    # img = viewer.layers['CT_SCAN'].data
    # rotate here is needed
    # TODO: this normal needs to be used to rotate the segmentation and the CT scan before taking the array at the point based on chosen axis
    chosen_slice = img[point]
    fetched_plane_index = point

    # viewer.add_image(chosen_slice, name="chosen_slice", colormap="gray", interpolation2d="bicubic")

    # Save the chosen values here into a JSON for future use if needed
    with open('data.json', 'r') as f:
        loaded_data = json.load(f)
    # Save info into json
    loaded_data["chosen_point"] = point
    loaded_data["chosen_normal"] = normal
    loaded_data["chosen_axis"] = chosen_axis
    with open('data.json', 'w') as f:
        json.dump(loaded_data, f)


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
    roti = rot(img, [0,0,1], 45)
    viewer.add_image(roti, name="rotated 45", colormap="gray", interpolation2d="bicubic")
    roti = rot(img, [0, 0, 1], 315)
    viewer.add_image(roti, name="rotated -45", colormap="gray", interpolation2d="bicubic")


# MAKE SEGMENTATION
@viewer.bind_key('s')
def on_press_s(viewer):
    global iterations
    iterations += 1
    get_segmentation(viewer)
    get_uncertainty_field(viewer)
    user_check(viewer)


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
