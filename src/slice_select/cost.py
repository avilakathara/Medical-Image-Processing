import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# returns the sum of the uncertainty values in the specified plane
#     uncertainty:  the uncertainty field, a 3D array
#               p:  a 3D point that lies on the plane
# rot_x and rot_y:  the x and y euler angle rotation of the normal of the plane
def cost(uncertainty, p, rot_x, rot_y):
    normal = rotate_vector(np.array([1, 0, 0]), rot_x, rot_y)
    size = uncertainty.shape
    coordinates = get_coordinates(size, p, normal)
    values = uncertainty[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]]

    result = np.sum(values)
    return result


def cost_using_normal(uncertainty, p, normal):
    size = uncertainty.shape
    coordinates = get_coordinates(size, p, normal)
    values = uncertainty[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]]

    result = np.sum(values)
    return result

# returns the 3D coordinates that lie in the specified plane
#    size:  the shape of the the uncertainty field
#       p:  a 3D point that lies on the plane
#  normal:  the normal that defines the plane
def get_coordinates(size, p, normal):
    largest_distance = int(np.linalg.norm(size))

    # create array containing its own 2d indices
    indices = np.arange(-largest_distance, largest_distance + 1)

    row_indices, col_indices = np.meshgrid(indices, indices, indexing='ij')

    normal, a, b = get_orthonormal_vectors(normal)
    coordinate_matrix = row_indices[:, :, np.newaxis] * a + col_indices[:, :, np.newaxis] * b
    coordinates = np.reshape(coordinate_matrix, (-1, 3)) + p
    coordinates = coordinates.astype(np.int16)
    mask = np.all((coordinates >= np.array([0, 0, 0])) & (coordinates < size), axis=1)
    filtered_coordinates = coordinates[mask]

    return filtered_coordinates


def get_coordinates_alt(size, p, normal):
    largest_distance = int(np.linalg.norm(size))

    # create array containing its own 2d indices
    indices = np.arange(-largest_distance, largest_distance + 1)

    row_indices, col_indices = np.meshgrid(indices, indices, indexing='ij')

    normal, a, b = get_orthonormal_vectors(normal)

    uv_indices = np.stack(np.meshgrid(indices, indices), axis=-1).reshape(-1, 2)

    coordinate_matrix = row_indices[:, :, np.newaxis] * a + col_indices[:, :, np.newaxis] * b
    coordinates = np.reshape(coordinate_matrix, (-1, 3)) + p
    coordinates = coordinates.astype(np.int16)
    mask = np.all((coordinates >= np.array([0, 0, 0])) & (coordinates < size), axis=1)
    filtered_coordinates = coordinates[mask]
    filtered_indices = uv_indices[mask]

    return filtered_coordinates, filtered_indices


# rotates a 3d vector by a specified x and y rotation using euler angles specified in degrees
def rotate_vector(vector, rot_x, rot_y):
    rotation = R.from_euler('XYZ', [rot_x, rot_y, 0], degrees=True)
    result = rotation.apply(vector)
    return result

# returns an orthonormal basis to the given vector
def get_orthonormal_vectors(v):
    v1 = v / np.linalg.norm(v)
    v2 = np.random.rand(3)  # Random vector
    v2 -= v2.dot(v1) * v1
    v2 /= np.linalg.norm(v2)
    v3 = np.cross(v1, v2)
    return v1, v2, v3


if __name__ == "__main__":
    # test_arr = np.load('test_data/slice_20.npy')
    #
    # costs = []
    # for i in range(0, 180):
    #     point = np.array([10, 10, 10])
    #     costs.append(cost(test_arr, point, 0, i))
    #
    # plt.plot(costs)
    # plt.show()

    size = (6,6,6)
    point = (3,3,3)
    normal = (1,0,0)
    get_coordinates_alt(size, point, normal)





    # a = np.array([1, 2, 3])
    # b = np.array([4, 5])
    #
    # # Use meshgrid to create the desired array
    # result = np.stack(np.meshgrid(a, b), axis=-1).reshape(-1, 2)
    #
    # print(result)