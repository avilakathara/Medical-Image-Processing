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
import matplotlib.pyplot as plt
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

        new_image = np.ones(slice.shape)*2
        # img = np.ones(slice.shape)
        # dilated = binary_dilation(slice,footprint=np.ones((3, 3)))
        background_start = (0,0)
        for i in range(len(slice)):
            if slice[background_start[0],background_start[1]] == 0:
                break
            for j in range(len(slice[0])):
                if slice[i,j] == 0:
                    background_start = (i,j)
                    break

        # img[flood(dilated,background_start)] = 2
        # img[dilated] = 0

        # check for regions where dilation fully removed all organ labels
        contours, hierarchy = cv.findContours(slice.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        for contour in contours:
            draw_contour = np.zeros(slice.shape)
            for point in contour:
                draw_contour[point[0][1], point[0][0]] = 1

            dilation = binary_dilation(draw_contour,footprint=np.ones((3, 3)))
            filled_background = np.ones(slice.shape)
            filled_background[flood(dilation,background_start)] = 2

            # img = filled_background
            # img[dilation] = 0
            # if np.count_nonzero(img==1) != 0:
            #     new_image[img==1] = 1
            #     new_image[img==0] = 0
            #     continue

            dilation = binary_dilation(draw_contour,footprint=np.ones((2, 2)))
            img = np.ones(slice.shape)
            img[flood(dilation,background_start)] = 2
            img[dilation] = 0

            if np.count_nonzero(img==1) != 0:
                new_image[filled_background==1] = 0
                new_image[img==1] = 1
                continue

            dilation = binary_dilation(draw_contour,footprint=np.ones((1, 1)))
            img = np.ones(slice.shape)
            img[flood(dilation,background_start)] = 2
            img[dilation] = 0

            if np.count_nonzero(img==1) != 0:
                new_image[filled_background==1] = 0
                new_image[img==1] = 1
                continue

            new_image[draw_contour==1] = 1


        seed_points[index] = new_image

    # return seed_points,None

    start_time = time.time()
    prob = random_walker(image, seed_points.astype(int), beta=0.1, mode='cg_j',tol=0.001, copy=False, return_full_prob=True)
    print(time.time() - start_time, "seconds")

    labels = np.zeros(image.shape).astype(int)
    labels[prob[0]>=0.5] = 1
    return labels,prob


