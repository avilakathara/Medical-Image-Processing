from pathlib import Path
import numpy as np
import SimpleITK as sitk

import tkinter as tk
from tkinter import filedialog

labels_to_ids = {
    "Background": 0,
    "BrainStem": 1,
    "Chiasm": 2,
    "Mandible": 3,
    "OpticNerve_L": 4,
    "OpticNerve_R": 5,
    "Parotid_L": 6,
    "Parotid_R": 7,
    "Submandibular_L": 8,
    "Submandibular_R": 9,
}
def convert_to_numpy(source_dir, destination_dir, subset=None):
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Provided source dir does not exist: {source_dir}")

    if not destination_dir.exists():
        print(f"Provided destination dir does not exist, creating it at {destination_dir}")
        destination_dir.mkdir()

    if subset is None:
        subset = []

    for p in source_dir.rglob("img.nrrd"):
        patient_base_path = p.parent
        patient_id = patient_base_path.stem

        if patient_id in subset or len(subset) == 0:
            print(f"[{patient_id}] Processing ...")

            img_path = patient_base_path.joinpath("img.nrrd")
            structure_paths = list(patient_base_path.joinpath("structures").rglob("*.nrrd"))
            img = sitk.ReadImage(img_path)
            img_arr = sitk.GetArrayFromImage(img)
            structures_arr = np.zeros(img_arr.shape)  # we save as labelmap
            for sp in structure_paths:
                structure_name = sp.stem
                structure_id = labels_to_ids[structure_name]
                structure = sitk.ReadImage(sp)
                structure_arr = sitk.GetArrayFromImage(structure)
                structures_arr += structure_arr * structure_id

            patient_dest_dir = destination_dir.joinpath(patient_id)
            print(f"[{patient_id}] Saving to {patient_dest_dir}")
            try:
                patient_dest_dir.mkdir()
                np.save(patient_dest_dir.joinpath("img.npy"), img_arr)
                np.save(patient_dest_dir.joinpath("structures.npy"), structures_arr)
            except:
                print("Data already exists")

            print(f"[{patient_id}] Done")

    print("Finished conversion of MICCAI dataset to numpy arrays")


# PROMPT TO SELECT FILE
window = tk.Tk()
window.withdraw()
dir_from = filedialog.askdirectory()
window.destroy()

window = tk.Tk()
window.withdraw()
dir_to = filedialog.askdirectory()
window.destroy()

print(dir_from)
print(dir_to)

convert_to_numpy(dir_from, dir_to)

#D:\RP\RPData
# convert_to_numpy(dir_from, dir_to, subset=['0522c0708', '0522c0195', '0522c0479'])
#convert_to_numpy("D:/RP/RPData" , "D:/RP/UsedData", subset=['0522c0708', '0522c0195', '0522c0479'])