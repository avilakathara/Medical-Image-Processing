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
    step_size = 1
    normal = []
    iterations = 40
    for i in range(0, iterations):
        # Generate random point that, together with a normal, defines a plane
        chosen_x = int(np.random.uniform(0, x))
        print("at iteration {} (out of {}) we choose x as {}".format(i, iterations, chosen_x))
        # run gradient descent, starting at ax
        current_x, current_uncertainty, _ = gradient_descent(uncertainty, chosen_x, step_size)
        # compare gradient result with current best result
        if highest_uncertainty < current_uncertainty:
            highest_uncertainty = current_uncertainty
            highest_x = current_x

    return uncertainty[highest_x], highest_x, normal, "x"

# Start pos is of type [x,y,z]
def gradient_descent(uncertainty, start_pos, step_size):
    m_x, m_y, m_z = uncertainty.shape
    current_pos = start_pos
    for i in range(40):
        # print("current position: " + str(current_pos))

        # prevent the current value from going outside of the allowed values
        # TODO: might be cleaner with np.clip
        if current_pos[0] <= 1:
            current_pos[0] = 2
        if current_pos[0] >= m_x - 1:
            current_pos[0] = m_x - 2
        if current_pos[1] <= 1:
            current_pos[1] = 2
        if current_pos[1] >= m_y - 1:
            current_pos[1] = m_y - 2
        if current_pos[2] <= 1:
            current_pos[2] = 2
        if current_pos[2] >= m_z - 1:
            current_pos[2] = m_z - 2

        # gradient_difference is the gradient above the current plane, gradient_difference_p is the gradient below the
        # current plane.
        # [current_pos + 1] because in get optimal slice we start from index 1, meaning everything is shifted by 1.
        gradient = get_point_gradient(uncertainty, current_pos, [1,0,0])

        print("current gradient: " + str(gradient))
        print("current point: " + str(current_pos))

        # TODO: Maybe make the step size reduction dynamic using i
        # gradient ascent
        mult = gradient * step_size
        current_pos += mult.astype(int)

    return current_pos, current_pos

# , x, normal, step_size
def get_point_gradient(uncertainty, point, normal):
    gradients = get_gradients(uncertainty)

    # Get the plane in relation to the x and normal
    # TODO: randomly instantiate the point and normal
    indexes = get_indexes_in_plane(uncertainty, point, normal, 0.00001)

    uncertainty_plane = uncertainty[indexes[:, 0], indexes[:, 1], indexes[:, 2]]
    gradient_plane = gradients[:,indexes[:, 0], indexes[:, 1], indexes[:, 2]]

    mult = gradient_plane * uncertainty_plane
    gradient_vector = np.sum(mult, axis=1)



    return gradient_vector

def get_plane_indices(uncertainty, point, normal):
    x, y, z = uncertainty.shape


def get_indexes_in_plane(array, position, normal, tolerance):
    A, B, C = normal
    D = -(A * position[0] + B * position[1] + C * position[2])  # Precompute the D constant

    min_i, max_i = 0, array.shape[0] - 1
    min_j, max_j = 0, array.shape[1] - 1
    min_k, max_k = 0, array.shape[2] - 1

    indexes_in_plane = []

    for i in range(min_i, max_i + 1):
        for j in range(min_j, max_j + 1):
            for k in range(min_k, max_k + 1):
                result = A*i + B*j + C*k + D

                if abs(result) <= tolerance:
                    indexes_in_plane.append([i, j, k])

    return np.array(indexes_in_plane)

def get_gradients(arr):
    gradients = np.gradient(arr, axis=None)
    return np.array(gradients)


def get_xygrad(plane):
    # cv2 here
    scale = 1
    delta = 0
    ddepth = cv2.CV_64F
    scaled_plane = np.round(plane * 255).astype(np.uint8)
    grad_x = cv2.Sobel(scaled_plane, ddepth, 1, 0, ksize=3)
    grad_y = cv2.Sobel(scaled_plane, ddepth, 0, 1, ksize=3)
    grad_x = (grad_x / 255.0).astype(np.float32)
    grad_y = (grad_y / 255.0).astype(np.float32)
    mult_x = abs(grad_x * plane)
    mult_y = abs(grad_y * plane)

    return mult_x, mult_y

    # sx = ndimage.sobel(plane, axis=0, mode='constant')
    # mx = abs(sx * plane)
    # #print(mx)
    # sy = ndimage.sobel(plane, axis=0, mode='constant')
    # my = abs(sy * plane)
    # #print(mx)
    # return mx, my


def get_grad(plane):
    x, y = get_xygrad(plane)
    return (np.sum(x) + np.sum(y))


def rotate(normal):
    # Compute the rotation angle and axis
    angle = np.arccos(normal[0])
    # Axis: (1,0,0) = (1,2), (0,1,0) = (0,2), (0,0,1) = (0,1)
    rotated_arr = np.transpose(np.roll(arr, -int(np.degrees(angle)), axis=(1, 2)), axes=(0, 2, 1))
    return rotated_arr

def get_orthonormal_vectors(v):
    v1 = v / np.linalg.norm(v)
    v2 = np.random.rand(3)  # Random vector
    v2 -= v2.dot(v1) * v1
    v2 /= np.linalg.norm(v2)
    v3 = np.cross(v1, v2)
    return v1, v2, v3

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
    # print("Mean values of planes: " + str(np.mean(test_arr, (1, 2))))
    # x = int(np.random.uniform(1, 19))
    #
    # # Random normal
    # normal = np.random.uniform(0, 1, 3)
    # normal = normal / np.sum(normal)
    # print(arr.shape)
    # Rotate the uncertainty according to normal
    # plane = arr[:, :, x] + np.outer(normal, np.arange(arr.shape[0])) + np.outer(np.arange(arr.shape[1]), normal)

    arr = []
    # ra = np.load('uncertainty.npy')

    result = gradient_descent(test_arr, [4, 10, 10], 0.1)
    print(result)


    # for x in range(1, len(ra)):
    #     arr.append(get_grad(ra[x]))
    #
    # print(arr)
    # plt.plot(arr)
    # plt.show()
    # #
    #
    # h_v, h_x, v, _ = get_optimal_slice(ra)
    # # print(h_v)
    # print(h_x)
    # print(v)
    # print("done")
    # print(test_arr[h_x + 1])
    # print(test_arr[h_x - 1])
    # print(np.sum(test_arr[1]))
    # print(np.sum(test_arr[4]))
