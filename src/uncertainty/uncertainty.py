
import math
import numpy as np

# takes the output from segment() and returns a 3D-array with uncertainties [0-1]
def calculate_uncertainty_fields(image, label, prob):

    # get image size
    ix, iy, iz = image.shape

    # assure other 3D-arrays have same size
    lx, ly, lz = label.shape
    if ix != lx or iy != ly or iz != lz:
        raise Exception("Inconsistent array sizes. Expected "
                        + "[" + ix + ", " + iy + ", " + iz + "]"
                        + "but got "
                        + "[" + lx + " ," + ly + ", " + lz + "].")
    px, py, pz = prob.shape
    if ix != px or iy != py or iz != pz:
        raise Exception("Inconsistent array sizes. Expected "
                        + "[" + ix + ", " + iy + ", " + iz + "]"
                        + "but got "
                        + "[" + px + " ," + py + ", " + pz + "].")

    # initialize output uncertainty field
    output_field = np.zeros((ix, iy, iz))

    # to be used for calculations
    mean_fg, std_fg, mean_bg, std_bg = gaussian_foreground_background(image, label)

    # calculate uncertainty for each voxel
    for x in range(0, ix):
        for y in range(0, iy):
            for z in range(0, iz):
                u_e = entropy_energy(prob[x, y, z])
                u_b = boundary_energy((x, y, z), label[x, y, z], image)
                u_r = regional_energy(image[x, y, z], label[x, y, z],
                                      mean_fg, std_fg, mean_bg, std_bg)
                output_field[x, y, z] = 0.8 * u_e + 0.05 * u_b + 0.15 * u_r

    return output_field

###

def entropy_energy(x_prob):
    t1 = x_prob * math.log(x_prob, 2)
    t2 = (1 - x_prob) * math.log(1 - x_prob, 2)
    return -t1 - t2

###

def boundary_energy(x_coord, label, image, alpha=2):
    delta = soft_delta_func(distance_to_boundary(x_coord, label))
    gradient_abs_pow = math.pow(abs(image_gradient(x_coord, image)), alpha)
    return delta / (1 + gradient_abs_pow)

def distance_to_boundary(x_coord, label):
    # TODO - import or transcribe MATLAB.bwdist
    distance_map = bwdist(label)

    # TODO - calculate distance
    return 0

def image_gradient(x_coord, image):
    # TODO - calculate image gradient
    return 0

def soft_delta_func(x):
    return math.pow(math.e, -(x * x) / 2)

###

def regional_energy(intensity, label, mean_fg, std_fg, mean_bg, std_bg):
    fg = gaussian_density(intensity, mean_fg, std_fg)
    bg = gaussian_density(intensity, mean_bg, std_bg)

    if label == 0:
        return gaussian_density(intensity, mean_bg, std_bg) / (fg + bg)
    return gaussian_density(intensity, mean_fg, std_fg) / (fg + bg)

def gaussian_foreground_background(image, label):
    ix, iy, iz = image.shape
    fg = []
    bg = []

    for x in range(0, ix):
        for y in range(0, iy):
            for z in range(0, iz):
                if label[x, y, z] == 0:
                    bg.append(image[x, y, z])
                if label[x, y, z] == 1:
                    fg.append(image[x, y, z])
                else:
                    continue

    return np.mean(fg), np.std(fg), np.mean(bg), np.std(bg)

def gaussian_density(x, mean, std):
    expo = -math.pow((x - mean) / std, 2) / 2
    coef = 1 / (std * math.sqrt(2 * math.pi))
    return coef * math.pow(math.e, expo)
