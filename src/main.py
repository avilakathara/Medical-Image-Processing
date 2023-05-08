import os
import napari
from napari.qt import thread_worker
from qtpy.QtWidgets import QMessageBox
#from src.segmentation.segmentation import segment
#from src.slice_select.optimization import get_optimal_slice
#from src.uncertainty.uncertainty import calculate_uncertainty_fields
from process_patients import *


#folder_path = "D:/RP/RPData/PDDCA-1.4.1_part1" # replace with the path of your folder
folders = ["D:/RP/RPData/PDDCA-1.4.1_part1", "D:/RP/RPData/PDDCA-1.4.1_part2", "D:/RP/RPData/PDDCA-1.4.1_part3"]
folder_names = []

# loop over all files in the folder
for folder_path in folders:
    for item in os.listdir(folder_path):
        # check if the item is a directory
        if os.path.isdir(os.path.join(folder_path, item)):
            folder_names.append(item)

def load_image(viewer, filename):
    patient_dest_dir = Path("D:/RP/UsedData/{}".format(filename))
    img = np.load(patient_dest_dir.joinpath("img.npy"))
    print(img.shape)
    print(img)
    structures = np.load(patient_dest_dir.joinpath("structures.npy")).astype(int)
    print(structures.shape)
    print(structures)
    img_layer = viewer.add_image(img, name="CT scan", colormap="gray", interpolation2d="bicubic")
    seg_layer = viewer.add_labels(structures, name="segmentation")
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
#convert_to_numpy("D:/RP/RPData" , "D:/RP/UsedData", subset=set)

# create a napari viewer
viewer = napari.Viewer()

# Choose the image we want to run the code on, and then load into viewer
filename = '0522c0708'
load_image(viewer, filename)


@viewer.bind_key('s')
def segment(viewer):
    new_filename = '0522c0195'
    load_image(viewer, new_filename)

@viewer.bind_key('y')
def user_check(viewer):
    confirmed = confirm_dialog("Process Status", "Is this image sufficiently segmented?")
    # load the new image if the user confirmed
    if confirmed:
        # TODO: So here the user has accepted the segemtnation, what do we do with this?
        #napari.quit()
        print("OVER")
    else:
        print("do slice selection")
        # find optimal slice
        # load the slice as a special imaage with a name n

@viewer.bind_key('c')
def user_input(viewer):
    # use this name n instead of 'CT scan here'
    layer = viewer.layers['CT scan']
    modified_img = layer.data
    # The modified slice has been gotten, we do something with this


# start the napari event loop
napari.run()

