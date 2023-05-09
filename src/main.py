import os
import napari
import json
from napari.qt import thread_worker
from qtpy.QtWidgets import QMessageBox
#from src.segmentation.segmentation import segment
#from src.slice_select.optimization import get_optimal_slice
#from src.uncertainty.uncertainty import calculate_uncertainty_fields
from process_patients import *


# #folder_path = "D:/RP/RPData/PDDCA-1.4.1_part1" # replace with the path of your folder
# folders = ["D:/RP/RPData/PDDCA-1.4.1_part1", "D:/RP/RPData/PDDCA-1.4.1_part2", "D:/RP/RPData/PDDCA-1.4.1_part3"]
# folder_names = []
#
# # loop over all files in the folder
# for folder_path in folders:
#     for item in os.listdir(folder_path):
#         # check if the item is a directory
#         if os.path.isdir(os.path.join(folder_path, item)):
#             folder_names.append(item)


def load_image(viewer, filename, name):
    patient_dest_dir = Path("D:/RP/UsedData/{}".format(filename))
    img = np.load(patient_dest_dir.joinpath("img.npy"))
    print(img.shape)
    print(img)
    structures = np.load(patient_dest_dir.joinpath("structures.npy")).astype(int)
    print(structures.shape)
    print(structures)
    img_layer = viewer.add_image(img, name="CT_scan_{}".format(name), colormap="gray", interpolation2d="bicubic")
    seg_layer = viewer.add_labels(structures, name="segmentation_{}".format(name))
    return (img_layer, seg_layer)

def confirm_dialog(title, message):
    msg_box = QMessageBox()
    msg_box.setWindowTitle(title)
    msg_box.setText(message)
    msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    msg_box.setDefaultButton(QMessageBox.No)
    return msg_box.exec_() == QMessageBox.Yes

# load 3D image (proxies)
# The subset here is manually input in the code, we could
set = ['0522c0708', '0522c0195', '0522c0479']
#convert_to_numpy("C:/Users/jonas/Documents/git/Medical-Image-Processing/RPData", "C:/Users/jonas/Documents/git/Medical-Image-Processing/UsedData", subset=set)

data = {'chosen_point': 0, 'chosen_normal': [0,0,0], 'chosen_axis' : 'x' , 'user_input': []}
with open('data.json', 'w') as f:
    json.dump(data, f)

# create a napari viewer
viewer = napari.Viewer()

# Choose the image we want to run the code on, and then load into viewer
filename = '0522c0708'
load_image(viewer, filename, "1")


@viewer.bind_key('s')
def segment(viewer):
    # TODO: should be doing this, cause we get labels back form the agorithm
    #seg_layer = viewer.add_labels(structures, name="segmentation")
    new_filename = '0522c0195'
    load_image(viewer, new_filename, "s")

count = 1

@viewer.bind_key('y')
def user_check(viewer):
    confirmed = confirm_dialog("Process Status", "Is this image sufficiently segmented?")
    # load the new image if the user confirmed
    if confirmed:
        # TODO: So here the user has accepted the segmentation, what do we do with this?
        #napari.quit()
        print("OVER")
    else:
        print("do slice selection")
        #uncertainty_field = calculate_uncertainty_fields()
        # find optimal slice
        #uncertainty, point, normal, chosen_axis = get_optimal_slice(uncertainty_field)
        # load the slice as a special image with a name n
        # TODO: this normal needs to be used to rotate the segmentation and the CT scan before taking the array at the point based on chosen axis

        # Get the image
        img = viewer.layers['CT_scan_1'].data
        # rotate here is needed

        # this is currently taking place
        val = 100

        chosen_slice = img[100]

        img_layer = viewer.add_image(chosen_slice, name="chosen_slice", colormap="gray", interpolation2d="bicubic")

        with open('data.json', 'r') as f:
            loaded_data = json.load(f)
        # Save info into json
        #loaded_data["chosen_point"] = point
        #loaded_data["chosen_normal"] = normal
        #loaded_data["chosen_axis"] = chosen_axis
        with open('data.json', 'w') as f:
            json.dump(loaded_data, f)


@viewer.bind_key('c')
def user_input(viewer):
    # use this name n instead of 'CT scan here'
    layer = viewer.layers['Points'].data
    #print(layer)
    user_input = []
    for val in layer:
        user_input.append(val[1:].tolist())

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


# start the napari event loop
napari.run()

