
import math
import numpy as np

# takes the output from segment() and returns a 3d-array with uncertainties [0-1]
def calculate_uncertainty_fields(image, label, prob):

    # TODO - assure that the input 3D arrays are of same size

    # TODO - get image size
    max_x = 0
    max_y = 0
    max_z = 0

    output_field = np.zeros((max_x, max_y, max_z))

    for x in range(0, max_x):
        for y in range(0, max_y):
            for z in range(0, max_z):
                u_e = entropy_energy(prob[x, y, z])
                u_b = boundary_energy((x, y, z), label[x, y, z], image)
                u_r = regional_energy()
                output_field[x, y, z] = 0.8 * u_e + 0.05 * u_b + 0.15 * u_r

    return output_field

def entropy_energy(x_prob):
    t1 = x_prob * math.log(x_prob, 2)
    t2 = (1 - x_prob) * math.log(1 - x_prob, 2)
    return -t1 - t2

def boundary_energy(x_coord, label, image, alpha=2):
    delta = soft_delta_func(distance_to_boundary(x_coord, label))
    gradient_abs_pow = math.pow(abs(image_gradient(x_coord, image)), alpha)
    return delta / (1 + gradient_abs_pow)

def regional_energy():
    return 0

def distance_to_boundary(x_coord, label):
    # TODO - import or transcribe MATLAB.bwdist
    distance_map = bwdist(label)

    # TODO - calculate distance
    return 0

def image_gradient(x_coord, image):
    # TODO - calculate image gradient
    return 0

def normal_dist_func(mean, deviation):
    # TODO
    return 0

def soft_delta_func(x):
    return math.pow(math.e, -(x * x) / 2)
