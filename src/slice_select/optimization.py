import math

import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import rotate

# takes the uncertainty field and returns the plane (slice) with the highest uncertainty
def optimization():
    return

def get_optimal_slice(uncertainty):
    x, _, _ = uncertainty.shape
    highest_uncertainty = -1
    highest_x = -1
    step_size = 0.03
    normal = []
    for i in range(0, 40):
        print(i)
        # Generate random point that, together with a normal, defines a plane
        x = int(np.random.uniform(0, x))

        # run gradient descent, starting at ax
        current_x, current_uncertainty, _ = gradient_descent(uncertainty, x, step_size)
        # compare gradient result with current best result
        if highest_uncertainty < current_uncertainty:
            highest_uncertainty = current_uncertainty
            highest_x = current_x

    return uncertainty[highest_x], highest_x, normal, "x"


def gradient_descent(uncertainty, x, step_size):
    m, _, _ = uncertainty.shape
    #print(x)
    current_pos = x
    for i in range(250):
        # print("current position: " + str(current_pos))

        # prevent the current value from going outside of the allowed values
        if current_pos <= 1:
            current_pos = 2
        if current_pos >= m - 1:
            current_pos = m - 2

        # gradient_difference is the gradient above the current plane, gradient_difference_p is the gradient below the
        # current plane.
        # [current_pos + 1] because in get optimal slice we start from index 1, meaning everything is shifted by 1.
        gradient_difference = get_grad(uncertainty[current_pos + 1, :, :]) - get_grad(uncertainty[current_pos - 1, :, :])

        # gradient ascent
        # TODO: Maybe make the step size reduction dynamic using i
        # print("current gradient: " + str(gradient_difference))
        current_pos += int(gradient_difference * step_size)

    return current_pos, get_grad(uncertainty[current_pos, :, :]), x

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
    print("Mean values of planes: " + str(np.mean(test_arr, (1, 2))))
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
