import numpy as np

folder_paths = []
for folder_path in folder_paths:
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
    uncertainty_field = calculate_uncertainty_fields(img, segmentation, probabilities)

