from pathlib import Path
import numpy as np

patient_dest_dir = Path("D:/RP/UsedData/0522c0708")

img = np.load(patient_dest_dir.joinpath("img.npy"))
structures = np.load(patient_dest_dir.joinpath("structures.npy")).astype(int)

print(f"Number of labels: {np.unique(structures).size}")
print(f"Labels: {np.unique(structures)}")

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
    napari.run()