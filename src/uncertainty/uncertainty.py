
import math
import numpy as np
import numpy.linalg as nl
import scipy.ndimage as sn
import scipy.stats as ss

from src.uncertainty.max_likelihood_estimate import max_likelihood_estimate

binary_labels = []

def calculate_uncertainty_fields(image, label, prob):
    """
    Method for calculating uncertainty fields

    @param image:   3D image of intensity values
    @param label:   3D image of segmentation (0 for background, 1 for organ)
    @param prob:    1D + 3D arrays containing classification probabilities of each voxels, for each labels

    @return:        3D uncertainty field for each voxels, for each labels
    """

    print("Calculating uncertainty fields...")

    # get image size; ps is the number of label types
    ps, px, py, pz = prob.shape
    assert ps == 2

    prob_fg = prob[0]
    prob_bg = prob[1]

    # assure other 3D-arrays have same size
    lx, ly, lz = label.shape
    if lx != px or ly != py or lz != pz:
        raise Exception("Inconsistent array sizes. Expected "
                        + "[" + px + ", " + py + ", " + pz + "]"
                        + "but got "
                        + "[" + lx + " ," + ly + ", " + lz + "].")
    ix, iy, iz = image.shape
    if ix != px or iy != py or iz != pz:
        raise Exception("Inconsistent array sizes. Expected "
                        + "[" + px + ", " + py + ", " + pz + "]"
                        + "but got "
                        + "[" + ix + " ," + iy + ", " + iz + "].")

    # parameters for foreground & background distributions
    mall, sall, mfg, sfg, ratio = gaussian_foreground_background(image, label)

    # distance maps (or transform)
    # (distance_from_0s, distance_from_1s)
    df0, df1 = get_distance_maps(label)

    # image gradient
    dx, dy, dz = np.gradient(image)

    # vectorize functions
    vec_entropy_e = np.vectorize(entropy_energy)
    vec_boundary_e = np.vectorize(boundary_energy)
    vec_regional_e = np.vectorize(regional_energy)

    # calculate uncertainty for each voxel
    u_e = vec_entropy_e(prob_fg)
    u_b = normalize_arr(vec_boundary_e(label, df0, df1, dx, dy, dz))
    u_r = normalize_arr(vec_regional_e(image, label, mall, sall, mfg, sfg, ratio))

    output_field = u_r

    return output_field

# --- ENTROPY ENERGY

def entropy_energy(prob):
    t1 = 0.0
    if prob > 0.0:
        t1 = float(prob * math.log(prob, 2))
    t2 = 0.0
    if prob < 1.0:
        t2 = float((1.0 - prob) * math.log(1.0 - prob, 2))
    return -t1 - t2

# --- REGIONAL ENERGY ---

def regional_energy(intensity, label, allm, alls, fgm, fgs, ratio):
    fg = gaussian_density(intensity, fgm, fgs)
    all = gaussian_density(intensity, allm, alls)

    return fg * ratio / all

def gaussian_foreground_background(image, label):
    intensities_all = image.flatten()
    intensities_fg = image[label == 1]

    allm, alls = max_likelihood_estimate(intensities_all)
    fgm, fgs = max_likelihood_estimate(intensities_fg)

    ratio = len(intensities_fg) / len(intensities_all)

    return allm, alls, fgm, fgs, ratio

def gaussian_density(x, mean, std):
    if x == 0:
        return 0

    expo = -math.pow((x - mean) / std, 2) / 2
    coef = 1 / (std * math.sqrt(2 * math.pi))
    return coef * math.pow(math.e, expo)

# --- BOUNDARY ENERGY ---

def boundary_energy(label, df0, df1, dx, dy, dz, alpha=1):
    coef = soft_delta_func(df1)
    if label == 1:
        coef = soft_delta_func(df0)

    grad_mag = nl.norm(np.array([dx, dy, dz]))

    return coef / (1 + math.pow(grad_mag, alpha))

def distance_to_boundary(x_coord, label, dm):
    x, y, z = x_coord
    df0, df1 = dm
    if label == 0:
        # voxel is classified as 0, get distance from nearest 1
        return df1[x, y, z]
    else:
        # voxel is classified as 1, get distance from nearest 0
        return df0[x, y, z]

def soft_delta_func(x, alpha=1):
    return math.pow(math.e, -(x * x) / (2 * alpha))

def get_distance_maps(label):
    label_inv = (~label.astype(bool)).astype(int)
    return sn.distance_transform_edt(label), sn.distance_transform_edt(label_inv)

# --- HELPER FUNCTIONS

def normalize_arr(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr
