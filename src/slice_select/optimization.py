import math

import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import rotate


# takes the uncertainty field and returns the plane (slice) with the highest uncertainty
def optimization():
    return


def random_plane_normal():
    # Generate three random values between -1 and 1
    random_values = np.random.uniform(-1, 1, 3)

    # Normalize the vector to obtain a unit normal
    normal = random_values / np.linalg.norm(random_values)

    return normal


# multiple gradient descent
def get_optimal_slice(uncertainty):
    x, y, z = uncertainty.shape

    gradients = get_gradients(uncertainty)

    highest_uncertainty = -1
    highest_point = [-10000, -10000, -10000]
    step_size = 0.1

    normal = [1, 0, 0]

    iterations = 1

    for i in range(0, iterations):
        # Generate random point that, together with a normal, defines a plane
        chosen_point = [int(np.random.uniform(0, x)), int(np.random.uniform(0, y)), int(np.random.uniform(0, z))]
        # Generate a normal
        # chosen_normal = random_plane_normal()
        chosen_normal = normal
        print("at iteration {} (out of {}) we choose x as {} and normal as {}".format(i, iterations, chosen_point,
                                                                                      chosen_normal))

        # run gradient descent, starting at a point and normal
        current_point, current_normal = gradient_descent(uncertainty, chosen_point, chosen_normal, step_size, gradients)

        # get uncertainty at resulting point
        plane_indexes = get_indexes_in_plane(uncertainty, current_point, normal, 0.00001)
        uncertainty_plane = uncertainty[plane_indexes[:, 0], plane_indexes[:, 1], plane_indexes[:, 2]]
        current_uncertainty = np.sum(uncertainty_plane)

        # compare gradient result with current best result
        if highest_uncertainty < current_uncertainty:
            highest_uncertainty = current_uncertainty
            highest_point = current_point

    return uncertainty[highest_point], highest_point, normal, "x"


# Start pos is of type [x,y,z]
def gradient_descent(uncertainty, start_pos, start_normal, step_size, gradients):
    m_x, m_y, m_z = uncertainty.shape
    current_pos = start_pos
    current_normal = start_normal

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

        # Update the point position based on equation 8
        gradient, indexes = get_point_gradient(uncertainty, current_pos, current_normal, gradients)

        print("current gradient: " + str(gradient))
        print("current point: " + str(current_pos))

        # TODO: Maybe make the step size reduction dynamic using i
        # gradient ascent
        mult = gradient * step_size
        current_pos += mult.astype(int)

        # Update the normal based on equation 7
        update_value = get_normal_update(current_normal, gradients, indexes)
        print(update_value)

    return current_pos, current_normal


def get_normal_update(normal, gradients, indexes):
    # Do u times Jacobian of a
    matrix_a = np.array([[-normal[0], -normal[1], -normal[2]], [0, 0, 0]])
    #res_a = matrix_a * indexes
    res_a = np.dot(matrix_a, indexes.T)

    # Do v times Jacobian of b
    matrix_b = np.array([[0, 0, 0], [-normal[0], -normal[1], -normal[2]]])
    #res_b = matrix_b * indexes
    res_b = np.dot(matrix_b, indexes.T)

    # Sum the two values above
    matrix_sum = res_a + res_b

    # get [x,y,z] gradient values from current plane
    gradient_planes = gradients[:, indexes[:, 0], indexes[:, 1], indexes[:, 2]]

    update_x = np.dot(matrix_sum, gradient_planes[0])
    update_y = np.dot(matrix_sum, gradient_planes[1])
    update_z = np.dot(matrix_sum, gradient_planes[2])

    normal_update = [update_x, update_y, update_z]
    return normal_update


# returns the gradient vector of the plane defined by the given point and normal
def get_point_gradient(uncertainty, point, current_normal, gradients):
    # Get the plane in relation to the x and normal
    indexes = get_indexes_in_plane(uncertainty, point, current_normal, 0.00001)

    gradient_plane = gradients[:, indexes[:, 0], indexes[:, 1], indexes[:, 2]]

    # sum up the planes to get three sums
    gradient_vector = np.sum(gradient_plane, axis=1)

    return gradient_vector, indexes


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
                result = A * i + B * j + C * k + D

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
    test_arr = np.load('test_data/my_array.npy')
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

    uncertainty_plane, highest_point, normal, _ = get_optimal_slice(test_arr)
    print(highest_point)

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
