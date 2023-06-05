import math
import numpy as np
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import map_coordinates


def pad_image(image):
    current_shape = image.shape
    max_axis = np.argmax(current_shape)

    # Find the largest size among all dimensions
    max_size = current_shape[max_axis]

    # Find the smallest value in the original image
    min_value = np.min(image)

    # Calculate the padding widths for each dimension
    pad_widths = []
    for current_dim in current_shape:
        pad_width = max(0, max_size - current_dim)
        pad_widths.append((pad_width // 2, pad_width - pad_width // 2))

    # Pad the image array using the smallest value as the constant
    padded_image = np.pad(image, pad_widths, mode='constant', constant_values=min_value)

    padding_widths = pad_widths
    original_shape = current_shape

    return padded_image, pad_widths, original_shape

def unpad_image(padded_image, pad_widths, original_shape):
    # Calculate the indices for unpadding
    unpad_indices = []
    for i in range(len(original_shape)):
        pad_width = pad_widths[i]
        unpad_indices.append(slice(pad_width[0], -pad_width[1] or None))

    # Unpad the image array using the calculated indices
    unpadded_image = padded_image[(*unpad_indices, ...)]

    return unpadded_image

def rotate(image, axis, angle_deg, cval=0.0):
    min_value = np.min(image)
    # Define the rotation axis and angle
    axis = np.array(axis)  # Example: Rotate around the X-axis

    # Convert the angle to radians
    angle_rad = np.radians(angle_deg)

    # Calculate the rotation matrix
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    u = axis / np.linalg.norm(axis)
    rotation_matrix = np.array([
        [cos_theta + u[0] * u[0] * (1 - cos_theta), u[0] * u[1] * (1 - cos_theta) - u[2] * sin_theta,
         u[0] * u[2] * (1 - cos_theta) + u[1] * sin_theta],
        [u[1] * u[0] * (1 - cos_theta) + u[2] * sin_theta, cos_theta + u[1] * u[1] * (1 - cos_theta),
         u[1] * u[2] * (1 - cos_theta) - u[0] * sin_theta],
        [u[2] * u[0] * (1 - cos_theta) - u[1] * sin_theta, u[2] * u[1] * (1 - cos_theta) + u[0] * sin_theta,
         cos_theta + u[2] * u[2] * (1 - cos_theta)]
    ])

    # Define the center point for rotation
    center = np.array(image.shape) / 2.0
    translation = center - np.dot(rotation_matrix, center)
    # Perform the 3D rotation
    rotated_image = affine_transform(image, rotation_matrix, offset=translation, order=3, mode='constant', cval=min_value)

    return rotated_image


def angle_between_vectors(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a ** 2 for a in vec1))
    magnitude2 = math.sqrt(sum(b ** 2 for b in vec2))
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    angle_rad = math.acos(cosine_angle)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def image_rotate_back_1(image, normal):
    axis = np.array([1, 0, 0])
    angle = angle_between_vectors(axis, normal)
    if angle > 5:
        rotated_image = rotate(image, axis, 360 - angle)
    axis = np.array([0, 1, 0])
    angle = angle_between_vectors(axis, normal)
    if angle > 5:
        rotated_image = rotate(rotated_image, axis, 360 - angle)
    axis = np.array([0, 0, 1])
    angle = angle_between_vectors(axis, normal)
    if angle > 5:
        rotated_image = rotate(rotated_image, axis, 360 - angle)

    return rotated_image


def image_rotate_1(image, normal):
    axis = np.array([1, 0, 0])
    angle = angle_between_vectors(axis, normal)
    if angle > 5:
        rotated_image = rotate(image, axis, angle)
    axis = np.array([0, 1, 0])
    angle = angle_between_vectors(axis, normal)
    if angle > 5:
        rotated_image = rotate(rotated_image, axis, angle)
    axis = np.array([0, 0, 1])
    angle = angle_between_vectors(axis, normal)
    if angle > 5:
        rotated_image = rotate(rotated_image, axis, angle)

    return rotated_image


def find_rotation(normal):
    # Normalize the normal vector
    normal = np.array(normal)
    normal /= np.linalg.norm(normal)

    # Check if the vectors are already parallel
    if np.allclose(normal, [1, 0, 0]):
        return np.zeros(3)  # No rotation needed

    # Compute the cosine of the angle between the vectors
    cosine_angle = np.dot(normal, [1, 0, 0])

    # Compute the angle in radians
    angle = math.acos(cosine_angle)

    # Determine the axis of rotation
    axis = np.cross(normal, [1, 0, 0])
    axis /= np.linalg.norm(axis)

    return axis * angle


def get_val(x):
    if x < 0:
        return 360 + x
    return x


def add_padding(image, pad_length):
    pad_width = ((pad_length, pad_length), (pad_length, pad_length), (pad_length, pad_length))
    min_value = np.min(image)
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=min_value)
    return padded_image


def remove_padding(padded_image, pad_length):
    pad_width = ((pad_length, pad_length), (pad_length, pad_length), (pad_length, pad_length))
    unpadded_image = padded_image[pad_width[0][0]:-pad_width[0][1], pad_width[1][0]:-pad_width[1][1],
                     pad_width[2][0]:-pad_width[2][1]]
    return unpadded_image


def true_img_rot(image, normal):
    #print(image.shape)
    image, pad_width, original_shape = pad_image(image)
    #print(image.shape)
    x, y, z = image.shape
    m = min(x, min(y, z))
    #image = add_padding(image, m)
    normal = np.array(normal).astype(float)
    rotations = find_rotation(normal)
    true_rotations = np.degrees(rotations)
    rotations = abs(true_rotations)
    #print(true_rotations)
    rotim = image
    if rotations[0] > 0:
        rotim = rotate(rotim, [1, 0, 0], get_val(true_rotations[0]), cval=0.0)
    if rotations[1] > 0:
        rotim = rotate(rotim, [0, 1, 0], get_val(true_rotations[1]), cval=0.0)
    if rotations[2] > 0:
        rotim = rotate(rotim, [0, 0, 1], get_val(true_rotations[2]), cval=0.0)

    #rotim = remove_padding(rotim, m)
    return rotim, pad_width, original_shape


def true_img_rot_back(image, normal, pad_width, original_shape):
    x, y, z = image.shape
    m = min(x, min(y, z))
    #image = add_padding(image, m)
    normal = np.array(normal).astype(float)
    rotations = find_rotation(normal)
    true_rotations = -1 * np.degrees(rotations)
    #print(true_rotations)
    rotations = abs(true_rotations)
    rotim = image
    if rotations[0] > 0:
        rotim = rotate(rotim, [1, 0, 0], get_val(true_rotations[0]), cval=0.0)
    if rotations[1] > 0:
        rotim = rotate(rotim, [0, 1, 0], get_val(true_rotations[1]), cval=0.0)
    if rotations[2] > 0:
        rotim = rotate(rotim, [0, 0, 1], get_val(true_rotations[2]), cval=0.0)
    #rotim = remove_padding(rotim, m)
    #print(original_shape)
    rotim = unpad_image(rotim, pad_width, original_shape)
    #print(rotim.shape)
    return rotim

# Rotates 3D image around image center
# INPUTS
#   array: 3D numpy array
#   orient: list of Euler angles (phi,psi,the)
# OUTPUT
#   arrayR: rotated 3D numpy array
# by E. Moebel, 2020
def rotate_array(array, orient):
    phi = orient[0]
    psi = orient[1]
    the = orient[2]

    # create meshgrid
    dim = array.shape
    ax = np.arange(dim[0])
    ay = np.arange(dim[1])
    az = np.arange(dim[2])
    coords = np.meshgrid(ax, ay, az)

    # stack the meshgrid to position vectors, center them around 0 by substracting dim/2
    xyz = np.vstack([coords[0].reshape(-1) - float(dim[0]) / 2,  # x coordinate, centered
                     coords[1].reshape(-1) - float(dim[1]) / 2,  # y coordinate, centered
                     coords[2].reshape(-1) - float(dim[2]) / 2])  # z coordinate, centered

    # create transformation matrix
    r = R.from_euler('zxz', [phi, psi, the], degrees=True)
    mat = r.as_matrix()

    # apply transformation
    transformed_xyz = np.dot(mat, xyz)

    # extract coordinates
    x = transformed_xyz[0, :] + float(dim[0]) / 2
    y = transformed_xyz[1, :] + float(dim[1]) / 2
    z = transformed_xyz[2, :] + float(dim[2]) / 2

    x = x.reshape((dim[1],dim[0],dim[2]))
    y = y.reshape((dim[1],dim[0],dim[2]))
    z = z.reshape((dim[1],dim[0],dim[2])) # reason for strange ordering: see next line

    # the coordinate system seems to be strange, it has to be ordered like this
    new_xyz = [y, x, z]

    # sample
    arrayR = map_coordinates(array, new_xyz, order=1)
    arrayR = np.transpose(arrayR, (1, 0, 2))

    return arrayR

# # Example usage
# normal_vector = [0, 0.1, 0]
# rotation = find_rotation(normal_vector)
# rotation = np.degrees(rotation)
# print("Rotation axis and angle:", rotation)
