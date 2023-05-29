import math
import numpy as np
from scipy.ndimage import affine_transform


def rotate(image, axis, angle_deg, cval=0.0):
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
    rotated_image = affine_transform(image, rotation_matrix, offset=translation, order=1, mode='constant', cval=cval)

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
    angle = angle_between_vectors(axis)
    if angle > 5:
        rotated_image = rotate(image, axis, angle)
    axis = np.array([0, 1, 0])
    angle = angle_between_vectors(axis)
    if angle > 5:
        rotated_image = rotate(rotated_image, axis, angle)
    axis = np.array([0, 0, 1])
    angle = angle_between_vectors(axis)
    if angle > 5:
        rotated_image = rotate(rotated_image, axis, angle)

    return rotated_image


