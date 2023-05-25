from pathlib import Path
from skimage.segmentation import random_walker
from skimage.segmentation import flood
# from skimage.segmentation import expand_labels
# from skimage.morphology import ball
from skimage.morphology import binary_dilation
# from skimage.morphology import binary_erosion
# from skimage.morphology import binary_opening

# from skimage.morphology import binary_closing
# from skimage.morphology import area_closing
# from skimage.filters import gaussian
import numpy as np
# import matplotlib.pyplot as plt
import time
import cv2 as cv

from datetime import datetime
now = datetime.now()

# FOR THE REST OF THE GROUP:
# conda install -c anaconda pyamg
# pip install chardet


import pyamg

def segment(image,seed_points):
    seed_points = seed_points.astype(int)
    if seed_points is None or np.count_nonzero(seed_points) == 0:
        print('there are no seedpoints provided!')
    else:
        print("Segmenting with {} seed points...".format(np.count_nonzero(seed_points)))

    for index,slice in enumerate(seed_points):
        # if index == 0 or index == len(seed_points)-1:
        #     seed_points[index] = np.ones(slice.shape)*2
        #     continue
        if np.count_nonzero(slice) == 0:
            continue
        img = np.ones(slice.shape)
        dilated = binary_dilation(slice,footprint=np.ones((3, 3)))
        img[flood(dilated,(0,0))] = 2
        img[dilated] = 0
        seed_points[index] = img

    start_time = time.time()
    prob = random_walker(image, seed_points, beta=0.1, mode='cg_j',tol=0.1, copy=False, return_full_prob=True)
    # print(time.time() - start_time, "seconds")

    labels = np.zeros(image.shape).astype(int)
    labels[prob[0]>=0.5] = 1

    return labels,prob


