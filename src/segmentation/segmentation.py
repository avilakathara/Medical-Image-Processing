from pathlib import Path
from skimage.segmentation import random_walker
import numpy as np

patient_dest_dir = Path("/Users/Bram/Documents/RP/miccai_data_numpy/part1/0522c0002")

img = np.load(patient_dest_dir.joinpath("img.npy"))
structures = np.load(patient_dest_dir.joinpath("structures.npy")).astype(int)
structures[structures != 3] = 0
print(len(img),len(img[0]),len(img[0,0]))

img = img[59:90,100:300,180:340]
structures = structures[59:90,100:300,180:340]

# one voxel as seed point for each label
# markers = [[],[74, 239, 260],[85, 217, 259],[65, 185, 232],[86, 197, 277],[86, 187, 241],[65, 217, 302],[64, 218, 219],[58, 201, 287],[57, 205, 236],[61, 209, 229]]
test = np.zeros(structures.shape).astype(int)
# for i in range(1,len(markers)-1):
#     test[markers[i][0],markers[i][1],markers[i][2]] = i
# test[markers[10][0],markers[10][1],markers[10][2]] = 16
seedpoints0522c0002_organ3 = [[4,14,65],[4,15,85],[5,27,54],[5,22,96],[6,65,37],[6,65,37],[6,40,110],[6,13,74]]
for point in seedpoints0522c0002_organ3:
    test[point[0],point[1],point[2]] = 3
    test[6,1,74] = 1
    test[6,20,49] = 1
    test[6,29,75] = 1
    # test[:,0,:] = 1
    # test[:,-1,:] = 1
    # test[:,:,0] = 1
    # test[:,:,-1] = 1
    # test[0,:,:] = 1
    # test[-1,:,:] = 1


# a = 50
# b = 150
# c = 200
# size = 100
labels = random_walker(img, test, beta=10, mode='cg_j', spacing = [3.938,1.000,1.000])
# img = img[a:a+size,b:b+size,c:c+size]
# structures = structures[a:a+size,b:b+size,c:c+size]
# print(f"Number of labels: {np.unique(structures).size}")
# print(f"Labels: {np.unique(structures)}")
use_napari = True

if not use_napari:
    import matplotlib.pyplot as plt

    slice_num = 100
    fig, ax = plt.subplots(layout="tight", figsize=(5, 5))
    ax.imshow(img[slice_num], cmap="gray")
    ax.imshow(structures[slice_num], alpha=(structures[slice_num] > 0).astype(float))
    ax.set_axis_off()
    plt.show()

else:
    import napari

    viewer = napari.Viewer()
    viewer.add_image(img, name="CT scan", colormap="gray", interpolation2d="bicubic")

    viewer.add_labels(structures, name="segmentation")
    viewer.add_labels(labels, name="result of rw")
    napari.run()

# takes an 3D image, returns a graph for random

#def process_input():
#    return

#def random_walker():
#    return

# takes a 3D image, returns the segmentation (labels, uncertainties, etc.)
def segment():
    return

