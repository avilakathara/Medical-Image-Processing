import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt


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
    point_step_size = 0.5
    normal_step_size = 0.02

    normal = [1, 0, 0]

    iterations = 1

    for i in range(0, iterations):
        # Generate random point that, together with a normal, defines a plane
        chosen_point = np.array(
            [int(np.random.uniform(0, x)), int(np.random.uniform(0, y)), int(np.random.uniform(0, z))])
        # chosen_point = np.array([10,10,10])
        # Generate a normal
        chosen_normal = random_plane_normal()
        # chosen_normal = normal
        print("at initialization {} (out of {}) we choose x as {} and normal as {}".format(i, iterations, chosen_point,
                                                                                           chosen_normal))

        # run gradient descent, starting at a point and normal
        current_point, current_normal = gradient_descent(uncertainty, chosen_point, chosen_normal, point_step_size,
                                                         normal_step_size, gradients)

        # get uncertainty at resulting point
        plane_indexes = get_indexes_in_plane(uncertainty, current_point, normal, 0.00001)
        print(current_point)
        print(plane_indexes.shape)
        uncertainty_plane = uncertainty[plane_indexes[:, 0], plane_indexes[:, 1], plane_indexes[:, 2]]
        current_uncertainty = np.sum(uncertainty_plane)

        # compare gradient result with current best result
        if highest_uncertainty < current_uncertainty:
            highest_uncertainty = current_uncertainty
            highest_point = current_point

    return uncertainty[highest_point], highest_point, normal, "x"


# Start pos is of type [x,y,z]
def gradient_descent(uncertainty, start_pos, start_normal, point_step_size, normal_step_size, gradients):
    m_x, m_y, m_z = uncertainty.shape
    current_pos = start_pos
    unrounded_pos = start_pos.astype(float)
    current_normal = start_normal

    for i in range(250):
        # Update the point position based on equation 8
        gradient, indexes, uv_indexes, a, b = get_point_gradient(uncertainty, current_pos, current_normal, gradients)

        # gradient ascent
        pos_update = gradient * point_step_size
        print(unrounded_pos)
        unrounded_pos += pos_update
        current_pos = unrounded_pos.astype(int)

        # prevent the current value from going outside of the allowed values
        # TODO: might be cleaner with np.clip
        if current_pos[0] <= 1:
            current_pos[0] = 2
            unrounded_pos[0] = 2
        if current_pos[0] >= m_x - 1:
            current_pos[0] = m_x - 2
            unrounded_pos[0] = m_x - 2
        if current_pos[1] <= 1:
            current_pos[1] = 2
            unrounded_pos[1] = 2
        if current_pos[1] >= m_y - 1:
            current_pos[1] = m_y - 2
            unrounded_pos[1] = m_y - 2
        if current_pos[2] <= 1:
            current_pos[2] = 2
            unrounded_pos[2] = 2
        if current_pos[2] >= m_z - 1:
            current_pos[2] = m_z - 2
            unrounded_pos[2] = m_z - 2

        # Update the normal based on equation 7
        normal_update = get_normal_update(current_normal, gradients, indexes, uv_indexes, a, b)

        current_normal += normal_step_size * normal_update
        current_normal /= np.linalg.norm(current_normal)

        print("current gradient: " + str(gradient))
        print("current point: " + str(current_pos))
        print("current normal: " + str(current_normal))

    return current_pos, current_normal


# indexes are the xyz coordinates of points that belong to the plane
def get_normal_update(normal, gradients, indexes, uv_indexes, a, b):
    # # Do u times Jacobian of a
    matrix_a = np.array([[-normal[0], -normal[1], -normal[2]], [0, 0, 0]])

    # # Do v times Jacobian of b
    matrix_b = np.array([[0, 0, 0], [-normal[0], -normal[1], -normal[2]]])

    # print("size of indexes {}".format(indexes.shape))

    sum = [0, 0]
    sum2 = [0, 0, 0]
    m, _ = uv_indexes.shape
    for i in range(m):
        uv = uv_indexes[i]
        u = uv[0]
        v = uv[1]
        res_a = matrix_a * u
        res_b = matrix_b * v
        matrix_sum = res_a + res_b
        sum2 += gradients[:, indexes[i][0], indexes[i][1], indexes[i][2]]
        res = np.dot(matrix_sum, gradients[:, indexes[i][0], indexes[i][1], indexes[i][2]])
        sum += res

    normal_update = a * sum[0] + b * sum[1]
    return normal_update


# returns the gradient vector of the plane defined by the given point and normal
def get_point_gradient(uncertainty, point, current_normal, gradients):
    # Get the plane in relation to the x and normal
    # indexes = get_indexes_in_plane(uncertainty, point, current_normal, 0.00001)
    uv_indexes, indexes, a, b = get_indices(current_normal, point, uncertainty)

    gradient_plane = gradients[:, indexes[:, 0], indexes[:, 1], indexes[:, 2]]

    # sum up the planes to get three sums
    gradient_vector = np.sum(gradient_plane, axis=1)

    return gradient_vector, indexes, uv_indexes, a, b


# Input: -a normal and reference point that define a plane
#        -a 3D array of uncertainties TODO: can be replaced by just its shape?
# Returns indices of
def get_indices(normal, point, uncertainty):
    # calculate normalized vectors orthogonal to the plane's normal
    normal, a, b = get_orthonormal_vectors(normal)
    # set the maximum values for a and b to iterate over
    # TODO the current max sizes are overestimates, can we make it smaller, is that even necessary?
    max_axis_size = np.max(uncertainty.shape) * 2
    # print("Max axis size: {}".format(max_axis_size))
    max_u = max_axis_size
    max_v = max_axis_size
    min_u = -max_axis_size
    min_v = -max_axis_size

    ab_array = []
    xyz_array = []
    # iterate over all possible a and b values
    for u in range(min_u, max_u):
        for v in range(min_v, max_v):
            # calculate the xyz coordinate as a linear combination of u and v, offset by the reference point
            xyz = point + (u * a + v * b).astype(np.int16)
            # check whether the calculated point falls within the uncertainty field
            if not (xyz[0] < 0 or xyz[1] < 0 or xyz[2] < 0 or xyz[0] >= uncertainty.shape[0] or
                    xyz[1] >= uncertainty.shape[1] or xyz[2] >= uncertainty.shape[2]):
                # matching (a,b) and (x,y,z) point will have the same index in the two arrays
                ab_array.append([u, v])
                xyz_array.append(xyz)
    return np.array(ab_array), np.array(xyz_array), a, b


def get_indexes_in_plane(array, position, normal, tolerance):
    A, B, C = normal
    D = -(A * position[0] + B * position[1] + C * position[2])  # Precompute the D constant

    min_i = 0
    max_i = array.shape[0] - 1
    min_j = 0
    max_j = array.shape[1] - 1
    min_k = 0
    max_k = array.shape[2] - 1

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


def get_grad(plane):
    x, y = get_xygrad(plane)
    return (np.sum(x) + np.sum(y))


def rotate(normal, arr):
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
    test_arr = np.load('test_data/sphere_20.npy')

    # ab_array, xyz_array = get_indices([1,-1,0], [7,10,10], test_arr)
    # print(ab_array)
    # print(xyz_array)

    # asdf = np.reshape(np.arange(5*5*5), (5,5,5))
    #
    # normal = [np.sqrt(2)/2,np.sqrt(2)/2,0]
    # rotated = rotate(unrotated,test_arr)
    # print("rotated array:")
    # print(rotated)
    # print("Mean values of planes: " + str(np.mean(test_arr, (1, 2))))
    # x = int(np.random.uniform(1, 19))
    #
    # # Random normal
    # normal = np.random.uniform(0, 1, 3)
    # normal = normal / np.sum(normal)
    # print(arr.shape)
    # Rotate the uncertainty according to normal
    # plane = arr[:, :, x] + np.outer(normal, np.arange(arr.shape[0])) + np.outer(np.arange(arr.shape[1]), normal)

    # arr = []

    print(test_arr.shape)
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
