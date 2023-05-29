import math
import numpy as np
from scipy.ndimage import affine_transform


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

    return padded_image


def rotate(image, axis, angle_deg, cval=0.0):
    image = pad_image(image)
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
    rotated_image = affine_transform(image, rotation_matrix, offset=translation, order=1, mode='constant', cval=min_value)

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
    print(image.shape)
    x, y, z = image.shape
    m = min(x, min(y, z))
    #image = add_padding(image, m)
    normal = np.array(normal).astype(float)
    rotations = find_rotation(normal)
    true_rotations = np.degrees(rotations)
    rotations = abs(true_rotations)
    rotim = image
    if rotations[0] > 0:
        rotim = rotate(rotim, [1, 0, 0], get_val(true_rotations[0]), cval=0.0)
    if rotations[1] > 0:
        rotim = rotate(rotim, [0, 1, 0], get_val(true_rotations[1]), cval=0.0)
    if rotations[2] > 0:
        rotim = rotate(rotim, [0, 0, 1], get_val(true_rotations[2]), cval=0.0)

    #rotim = remove_padding(rotim, m)
    return rotim


def true_img_rot_back(image, normal):
    x, y, z = image.shape
    m = min(x, min(y, z))
    #image = add_padding(image, m)
    normal = np.array(normal).astype(float)
    rotations = find_rotation(normal)
    true_rotations = -1 * np.degrees(rotations)
    print(true_rotations)
    rotations = abs(true_rotations)
    rotim = image
    if rotations[0] > 0:
        rotim = rotate(rotim, [1, 0, 0], get_val(true_rotations[0]), cval=0.0)
    if rotations[1] > 0:
        rotim = rotate(rotim, [0, 1, 0], get_val(true_rotations[1]), cval=0.0)
    if rotations[2] > 0:
        rotim = rotate(rotim, [0, 0, 1], get_val(true_rotations[2]), cval=0.0)

    #rotim = remove_padding(rotim, m)
    return rotim

# # Example usage
# normal_vector = [0, 0.1, 0]
# rotation = find_rotation(normal_vector)
# rotation = np.degrees(rotation)
# print("Rotation axis and angle:", rotation)
