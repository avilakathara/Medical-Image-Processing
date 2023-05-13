import math

import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import rotate

# takes the uncertainty field and returns the plane (slice) with the highest uncertainty


best_of_planes = []


# Input: 3D numpy array
# Output: a 2D array representing the indexes in the 3D array

# def get_optimal_slice():
#     return
#
#
# def multi_gradient_descent(uncertainty, iterations, step_size, multi):
#     highest_value = -1
#     for i in range(0, multi):
#         normal = np.random.rand(3) * 360
#         temp = gradient_descent(uncertainty, iterations, step_size, normal)
#         if highest_value < temp:
#             highest_value = temp
#     return highest_value
#
#
# # Input: a 3D numpy array with values between 0 and 1
# def gradient_descent(uncertainty, iterations, step_size, normal):
#     # instantiate random plane: random position within current segmentation
#     current_pos = 56
#     for i in range(iterations):
#         # calculate gradient (uncertainty of one step above - one step below)
#         gradient = calculate(current_pos+1) - calculate(current_pos-1)
#         current_pos += gradient * step_size  # move to higher uncertainty, repeat
#
#     uncertainty_value = calculate(current_pos)
#
#     return uncertainty_value, current_pos
#
#
# # Sum over a 2d array
# def calculate(pos):
#     return 0

def optimization():
    return

def get_optimal_slice(uncertainty):
    x, _, _ = uncertainty.shape
    highest_value = -1
    highest_x = -1
    step_size = 10
    v = -1
    normal = []
    for i in range(0, 10):
        # Random Point on the plane
        x = int(np.random.uniform(1, x - 3))

        # # Random normal
        # normal = np.random.uniform(0, 1, 3)
        # normal = normal / np.sum(normal)
        #
        # # Rotate the uncertainity according to normal
        # plane = arr[:, :, x] + np.outer(normal, np.arange(arr.shape[0])) + np.outer(np.arange(arr.shape[1]), normal)

        temp_x, temp, temp_v = gradient_descent3(uncertainty, x, step_size)
        if highest_value < temp:
            highest_value = temp
            highest_x = temp_x
            v = temp_v

    return uncertainty[highest_x], highest_x, normal, "x"
    #return highest_value, highest_x, v

# Currently using a very simple step size update i.e reduce over time
def gradient_descent2(uncertainty, x, step_size):
    m, _, _ = uncertainty.shape
    #print(x)
    current_pos = x
    for i in range(250):
        #print(current_pos)
        gradient_difference = func(uncertainty[current_pos + 2, :, :]) - func(uncertainty[current_pos + 1, :, :])

        gradient_difference_p = func(uncertainty[current_pos + 1, :, :]) - func(uncertainty[current_pos, :, :])

        if gradient_difference >= 0 and gradient_difference_p >= 0:
            if gradient_difference < gradient_difference_p:
                step_size = int(step_size / 2)
        if gradient_difference < 0 and gradient_difference_p < 0:
            if abs(gradient_difference_p) < abs(gradient_difference):
                step_size = int(step_size / 2)
        if gradient_difference <= 0 and gradient_difference_p >= 0:
            step_size = 1
        if gradient_difference >= 0 and gradient_difference_p <= 0:
            step_size = 1
        #print(gradient_difference)
        if gradient_difference > 0:
            current_pos += step_size  # difference in position should or shouldn't be dependant on gradient?
            if current_pos >= m:
                current_pos = m-3
        else:
            current_pos -= step_size
            if current_pos <= 0:
                current_pos = 2
    return current_pos + 1, func(uncertainty[current_pos + 1, :, :]), x

def gradient_descent3(uncertainty, x, step_size):
    m, _, _ = uncertainty.shape
    #print(x)
    current_pos = x
    for i in range(250):
        #print(current_pos)
        gradient_difference = get_grad(uncertainty[current_pos + 2, :, :]) - get_grad(uncertainty[current_pos + 1, :, :])

        gradient_difference_p = get_grad(uncertainty[current_pos + 1, :, :]) - get_grad(uncertainty[current_pos, :, :])

        if gradient_difference >= 0 and gradient_difference_p >= 0:
            if gradient_difference < gradient_difference_p:
                step_size = int(step_size / 2)
        if gradient_difference < 0 and gradient_difference_p < 0:
            if abs(gradient_difference_p) < abs(gradient_difference):
                step_size = int(step_size / 2)
        if gradient_difference <= 0 and gradient_difference_p >= 0:
            step_size = 1
        if gradient_difference >= 0 and gradient_difference_p <= 0:
            step_size = 1
        #print(gradient_difference)
        if gradient_difference > 0:
            current_pos += step_size  # difference in position should or shouldn't be dependant on gradient?
            if current_pos >= m:
                current_pos = m-3
        else:
            current_pos -= step_size
            if current_pos <= 0:
                current_pos = 2
    return current_pos + 1, get_grad(uncertainty[current_pos + 1, :, :]), x

def func(plane):
    gradient_image = cv2.Laplacian(plane, cv2.CV_64F)
    arr = abs(gradient_image * plane)
    return np.sum(arr)

def get_xygrad(plane):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    #grad_x = cv2.Sobel(plane, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    #grad_y = cv2.Sobel(plane, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    #mult_x = abs(grad_x * plane)

    sx = ndimage.sobel(plane, axis=0, mode='constant')
    mx = abs(sx * plane)
    #print(mx)
    sy = ndimage.sobel(plane, axis=0, mode='constant')
    my = abs(sy * plane)
    #print(mx)
    return mx, my

def get_grad(plane):
    x,y = get_xygrad(plane)
    return (np.sum(x) + np.sum(y))

def rotate(normal):
    # Compute the rotation angle and axis
    angle = np.arccos(normal[0])
    # Axis: (1,0,0) = (1,2), (0,1,0) = (0,2), (0,0,1) = (0,1)
    rotated_arr = np.transpose(np.roll(arr, -int(np.degrees(angle)), axis=(1, 2)), axes=(0, 2, 1))
    return rotated_arr

if __name__ == "__main__":
    # arr = np.arange(20)
    # arr = arr.reshape((-1,1,1))
    # arr_2d = np.repeat(arr, 20,axis=1)
    # arr_3d = np.repeat(arr_2d, 20,axis=2)
    # test_arr = 1 - np.abs(arr_3d/10 - 1)
    # noise = np.random.rand(20,20,20)/10
    # test_arr = test_arr + noise
    # np.save('my_array.npy', test_arr)
    test_arr = np.load('my_array.npy')
    arr = np.load('my_array.npy')
    print("Mean: " + str(np.mean(test_arr, (1, 2))))
    # x = int(np.random.uniform(1, 19))
    #
    # # Random normal
    # normal = np.random.uniform(0, 1, 3)
    # normal = normal / np.sum(normal)
    # print(arr.shape)
    # Rotate the uncertainty according to normal
    #plane = arr[:, :, x] + np.outer(normal, np.arange(arr.shape[0])) + np.outer(np.arange(arr.shape[1]), normal)

    arr = []

    for x in range(1, 20):
        arr.append(get_grad(test_arr[x]))

    print(arr)
    plt.plot(arr)
    plt.show()
    #
    h_v, h_x, v, _ = get_optimal_slice(test_arr)
    #print(h_v)
    print(h_x)
    print(v)
    print("done")
    # print(test_arr[h_x + 1])
    # print(test_arr[h_x - 1])
    # print(np.sum(test_arr[1]))
    # print(np.sum(test_arr[4]))
