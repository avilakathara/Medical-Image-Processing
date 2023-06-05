import math

import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt

# from scipy import ndimage
# from scipy.ndimage import rotate
from slice_select.rotation_methods import rotate

from slice_select.rotation_methods import true_img_rot


def discreet_get_optimal_slice(uncertainty, x_axis=False, y_axis=False, z_axis=False, diagonal_1=False,
                               diagonal_2=False, diagonal_3=False, diagonal_4=False, diagonal_5=False,
                               diagonal_6=False):
    print("Fetching most uncertain slice...")

    current_maximum_uncertainity = -1
    current_point = 0
    chosen_axis = "none"
    show_plot = False

    # Compute the rotated uncertainty fields here so the time isn't added to the choice part of the function

    # Rotate around z axis by 45 degrees.
    # d1_rotation = rotate(uncertainty, [0,0,1], 45)
    d1_rotation, _, _ = true_img_rot(uncertainty, [0.707, 0.707, 0])
    # Rotate around z axis by -45 degrees.
    d2_rotation = rotate(uncertainty, [0, 0, 1], 315)
    # Rotate around y axis by 45 degrees.
    d3_rotation = rotate(uncertainty, [0, 1, 0], 45)
    # Rotate around y axis by -45 degrees.
    d4_rotation = rotate(uncertainty, [0, 1, 0], 315)
    # Rotate around x axis by 45 degrees.
    d5_rotation = rotate(uncertainty, [1, 0, 0], 45)
    # Rotate around x axis by -45 degrees.
    d6_rotation = rotate(uncertainty, [1, 0, 0], 315)

    # gradients = get_gradients(uncertainty)
    gradients = np.array([uncertainty, uncertainty, uncertainty])
    # print("gradients shape is {}".format(gradients.shape))

    if x_axis:
        sum_x, point = find_maximal_x(gradients, show_plot)
        if sum_x > current_maximum_uncertainity:
            current_maximum_uncertainity = sum_x
            chosen_axis = "x"
            current_point = point
    if y_axis:
        sum_y, point = find_maximal_y(gradients, show_plot)
        if sum_y > current_maximum_uncertainity:
            current_maximum_uncertainity = sum_y
            chosen_axis = "y"
            current_point = point
    if z_axis:
        sum_z, point = find_maximal_z(gradients, show_plot)
        if sum_z > current_maximum_uncertainity:
            current_maximum_uncertainity = sum_z
            chosen_axis = "z"
            current_point = point
    if diagonal_1:
        sum_t, point = find_maximal_x(np.array([d1_rotation, d1_rotation, d1_rotation]), show_plot)
        if sum_t > current_maximum_uncertainity:
            current_maximum_uncertainity = sum_t
            chosen_axis = "d1"
            current_point = point

    if chosen_axis == "x":
        return current_maximum_uncertainity, current_point, np.array([1, 0, 0]), "x"
    elif chosen_axis == "y":
        return current_maximum_uncertainity, current_point, np.array([0, 1, 0]), "y"
    elif chosen_axis == "z":
        return current_maximum_uncertainity, current_point, np.array([0, 0, 1]), "z"
    elif chosen_axis == "d1":
        return current_maximum_uncertainity, current_point, np.array([0.707, 0.707, 0]), "d1"
    # uncertainty, point, normal, chosen_axis


def find_maximal_x(gradients, graph):
    gradients = gradients[0]
    x, _, _ = gradients.shape

    max_sum = -1
    current_point = -1
    arr = []
    for i in range(x):
        t_sum = np.sum(gradients[i])
        arr.append(t_sum)
        if abs(t_sum) > max_sum:
            max_sum = abs(t_sum)
            current_point = i

    if graph:
        plt.title("x-axis")
        plt.plot(arr)
        plt.show()

    return max_sum, current_point


def find_maximal_y(gradients, graph):
    gradients = gradients[1]
    _, y, _ = gradients.shape

    max_sum = -1
    current_point = -1
    arr = []

    for i in range(y):
        t_sum = np.sum(gradients[:, i, :])
        arr.append(t_sum)
        if abs(t_sum) > max_sum:
            max_sum = abs(t_sum)
            current_point = i
    if graph:
        plt.title("y-axis")
        plt.plot(arr)
        plt.show()

    return max_sum, current_point


def find_maximal_z(gradients, graph):
    gradients = gradients[2]
    _, _, z = gradients.shape

    max_sum = -1
    current_point = -1
    arr = []

    for i in range(z):
        t_sum = np.sum(gradients[:, :, i])
        arr.append(t_sum)
        if abs(t_sum) > max_sum:
            max_sum = abs(t_sum)
            current_point = i
    if graph:
        plt.title("z-axis")
        plt.plot(arr)
        plt.show()

    return max_sum, current_point


# Get gradients of each point in the x, y and z directions
def get_gradients(arr):
    gradients = np.gradient(arr, axis=None)
    return np.array(gradients)


if __name__ == "__main__":
    print("Discreet optimization")
    test_arr = np.load('uncertainty.npy')
    print(test_arr.shape)
    # uncertainty, point, normal, chosen_axis
    m, p, n, c = discreet_get_optimal_slice(test_arr, True, True, True, True)
    print("chosen axis is {} with normal {} with maximum value of {} at point {}".format(c, n, m, p))
