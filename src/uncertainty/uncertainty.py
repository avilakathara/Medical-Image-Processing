
import math
import numpy as np
import numpy.linalg as nl
import scipy.ndimage as sn

def calculate_uncertainty_fields(image, label, prob):
    """
    Method for calculating uncertainty fields

    @param image:   3D image of intensity values
    @param label:   3D image of segmentation (0 for background, 1 for organ 1, etc.)
    @param prob:    3D + 1D arrays containing classification probabilities of each voxels, for each labels
    @return:        3D + 1D arrays containing uncertainty values of each voxels, for each labels
    """

    # get image size; ps is the number of label types
    ps, px, py, pz = prob.shape

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

    # initialize output uncertainty field
    output_field = np.zeros((px, py, pz, ps))

    # parameters for foreground & background distributions
    # [(mean_0, std_0), (mean_1, std_1), ...]
    distributions = gaussian_foreground_background(image, label, ps)

    # distance maps (or transform)
    # [(distance_from_0, distance_from_1), (df0, df2), ...]
    distance_maps = get_distance_maps(label, ps)

    # image gradient
    dx, dy, dz = np.gradient(image)

    # calculate uncertainty for each voxel
    for x in range(px):
        for y in range(py):
            for z in range(pz):
                for p in range(1, ps):
                    u_e = entropy_energy(prob[p, x, y, z])
                    u_b = boundary_energy((x, y, z), label[x, y, z],
                                          distance_maps[p], (dx, dy, dz))
                    u_r = regional_energy(image[x, y, z], label[x, y, z],
                                          distributions[0], distributions[p])
                    output_field[x, y, z, p] = 0.8 * u_e + 0.05 * u_b + 0.15 * u_r

    return output_field

###

def entropy_energy(prob):
    t1 = prob * math.log(prob, 2)
    t2 = (1 - prob) * math.log(1 - prob, 2)
    return -t1 - t2

###

def boundary_energy(x_coord, label, dm, gradient, alpha=2):
    delta = soft_delta_func(distance_to_boundary(x_coord, label, dm))
    gradient_abs_pow = math.pow(voxel_norm(x_coord, gradient), alpha)
    return delta / (1 + gradient_abs_pow)

def distance_to_boundary(x_coord, label, dm):
    x, y, z = x_coord
    df0, df1 = dm
    if label == 0:
        # voxel is classified as 0, get distance from nearest 1
        return df1[x, y, z]
    else:
        # voxel is classified as 1, get distance from nearest 0
        return df0[x, y, z]

def voxel_norm(x_coord, gradient):
    x, y, z = x_coord
    dx, dy, dz = gradient
    return nl.norm(np.array([dx[x, y, z], dy[x, y, z], dz[x, y, z]]))

def soft_delta_func(x):
    return math.pow(math.e, -(x * x) / 2)

def get_distance_maps(label, n):
    lx, ly, lz = label.shape
    labels = []
    for i in range(1, n):
        labels.append(np.zeros((lx, ly, lz)))

    for x in range(lx):
        for y in range(ly):
            for z in range(lz):
                for i in range(1, n):
                    if label[x, y, z] == i:
                        labels[i][x, y, z] = 1
                    else:
                        labels[i][x, y, z] = 0

    output_map = []
    for i in range(1, n):
        output_map.append(0)
    for i in range(1, n):
        output_map.append((
            sn.distance_transform_edt(labels[i]),
            sn.distance_transform_edt(np.invert(labels[i]))
        ))
    return output_map

###

def regional_energy(intensity, label, dist_0, dist_1):
    mean_bg, std_bg = dist_0
    mean_fg, std_fg = dist_1
    fg = gaussian_density(intensity, mean_fg, std_fg)
    bg = gaussian_density(intensity, mean_bg, std_bg)

    if label == 0:
        return gaussian_density(intensity, mean_bg, std_bg) / (fg + bg)
    return gaussian_density(intensity, mean_fg, std_fg) / (fg + bg)

def gaussian_foreground_background(image, label, n):
    ix, iy, iz = image.shape
    intensity_list = []
    for i in range(n):
        intensity_list.append([])

    for x in range(ix):
        for y in range(iy):
            for z in range(iz):
                for i in range(n):
                    if label[x, y, z] == i:
                        intensity_list[i].append(image[x, y, z])
                    else:
                        continue

    output_list = []
    for i in range(n):
        output_list.append((np.mean(intensity_list[i]), np.std(intensity_list[i])))
    return output_list

def gaussian_density(x, mean, std):
    expo = -math.pow((x - mean) / std, 2) / 2
    coef = 1 / (std * math.sqrt(2 * math.pi))
    return coef * math.pow(math.e, expo)
